from pathlib import Path
import shutil
import argparse
from joblib import Parallel, delayed

def extrac_steps_from_json(stepDir, Id, outDir):
    chunk, filestem = Id.split('/')
    step_folder = stepDir / 'abc_{}_step_v00'.format(chunk) / filestem
    step_path = list(step_folder.glob('*.step'))[0]
    out_chunk = outDir / chunk
    if not out_chunk.exists():
        out_chunk.mkdir(parents=True, exist_ok=True)
    out_path = outDir / chunk /  (filestem + '.step')
    shutil.copyfile(str(step_path), str(out_path))

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--step_path',required=True,help='step dir to extract')
    parse.add_argument('--json_path',required=True,help='json dir to guide the step extraction')
    parse.add_argument('--out_path',required=True,help='output dir to store the step file')
    args = parse.parse_args()
    step_path = Path(args.step_path)
    json_path = Path(args.json_path)
    out_path = Path(args.out_path)
    jsonList = list(json_path.glob('*/*'))
    IdList = [(str(i).split('/')[-2] + '/' + i.stem) for i in jsonList]
    # extrac_steps_from_json(step_path, IdList[0], out_path)
    Parallel(n_jobs=10,verbose=2)(delayed(extrac_steps_from_json)(step_path,id, out_path) for id in IdList)