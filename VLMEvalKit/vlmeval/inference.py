import torch
import pandas as pd
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    # structs = [dataset.build_prompt(data.iloc[i]) for i in range(lt)]

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, model_dict_override, out_file, verbose=False, api_nproc=4, caption_file=None, qa_file=None, suffix=""):

    if caption_file:
        cap_data = load(caption_file)
        assert(cap_data['index'].is_unique)
        cap_data.set_index('index', inplace=True)
        captions_dict = cap_data['prediction'].to_dict()


    if qa_file:
        qa_data = load(qa_file)
        assert(qa_data['index'].is_unique)
        qa_data.set_index('index', inplace=True)
        qa_dict = qa_data['prediction'].to_dict()


    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}{suffix}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))
    # res = {}

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]
        
    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    if isinstance(model, str):
        model_partial = supported_VLM[model_name]
        model_func = model_partial.func
        model_args = model_partial.keywords
        model_args.update(model_dict_override)
        from functools import partial
        model_partial_new = partial(model_func, **model_args)
        print(model_partial)
        model = model_partial_new()
    else:
        model = model
    # model = supported_VLM[model_name]() if isinstance(model, str) else model

    # is_api = getattr(model, 'is_api', False)
    # if is_api:
    #     lt, indices = len(data), list(data['index'])
    #     supp = infer_data_api(
    #         model=model,
    #         work_dir=work_dir,
    #         model_name=model_name,
    #         dataset=dataset,
    #         index_set=set(indices),
    #         api_nproc=api_nproc)
    #     for idx in indices:
    #         assert idx in supp
    #     res.update(supp)
    #     res = {k: res[k] for k in data_indices}
    #     dump(res, out_file)
    #     return model
    # else:
    #     model.set_dump_image(dataset.dump_image)

    model.set_dump_image(dataset.dump_image)

    vllm_inputs = []
    indexes = []
    structs = []

    prompt_list = []
    pixel_value_list = []
    patch_list_list = []
    response_list = []

    # lt = 20
    for i in tqdm(range(lt)):

        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])
        #breakpoint()

    
        item = model._prepare_input(
            message=struct, dataset=dataset_name, 
            caption=captions_dict[idx] if caption_file else None,
            qa=qa_dict[idx] if qa_file else None
        )
        vllm_inputs.append(item)

        indexes.append(idx)
        structs.append(struct)

    outputs = model.llm.generate(
        vllm_inputs,
        sampling_params=model.sampling_params,
    )
    responses = [item.outputs[0].text for item in outputs]

    for idx, resp in zip(indexes, responses):
        res[idx] = resp


    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(model, work_dir, model_name, dataset, model_dict_override, verbose=False, api_nproc=4, ignore_failed=False, caption_file=None, qa_file=None, suffix=""):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}{suffix}.csv')
    print(result_file)
    prev_file = f'{work_dir}/{model_name}_{dataset_name}{suffix}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}{suffix}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset, model_dict_override=model_dict_override,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc, caption_file=caption_file, qa_file=qa_file, suffix=suffix)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model
