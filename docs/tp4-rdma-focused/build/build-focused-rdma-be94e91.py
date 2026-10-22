"""Integrator-run real CUDA build; preparation itself executes no remote command.

Requires an extracted final recipe in ROOT/context-v2, plus the pinned AOT.
Full source verification precedes a fresh native/probe/NVCC compile.
No GPU request, model load, proxy or lifecycle action is performed.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

NATIVE_COMMIT = 'be94e919e35c628261bd38364b1743a2e4353a26'
PATCH_SHA256 = '46111e72fa8432f5477a935a326ab4d8c7122cb7e3d4b29998e5791bb4d6dab2'
FIRST_THIRTEEN = {'0001-zig-shared-engine.patch': '923402e1ec409e43c2d31b59df64559a216b894dc3dfa11eb5f0cc6373f35222', '0002-zig-flashnext-cuda.patch': '3220b61b295745446421e7c7bc5a0146c3c1dcfa653af4afbcd6df6caf770ffb', '0003-zig-glm-kernels.patch': 'b6fd1d791dbca19e428186a840ba3973ba0b1d045f30098ade0245b163e78ff2', '0004-zig-glm-engine.patch': '685054c5237f5348f7734a151c86f0073df3e8d1ebbab51c6b122f166763247c', '0005-zig-glm-tests.patch': '9d0e969f4535ab5681ab3ca30945b7ef3ba99fc8a21fa6c250b1b4adf7d4401e', '0006-zig-glm-scores-rows-g128.patch': 'b426d1a5cd9009a33abb9dfbc4c3f5d357a47be53b34a51145870d6c65852c51', '0007-zig-flashnext-record-before-long-batch.patch': 'f1a7198f4a52c8eff35d75c1119912d5beb18b69b8eb067f1b7c490ec641333f', '0008-zig-aot-log-missing-variant.patch': 'cecf342ffbb6453640c5e1e16f5191988df120fe557d39fb9bf70480b1d1136b', '0009-zig-glm-tpn-pool.patch': '33097fb37b714737c327d398e635bf8675bc499f824e2ae02b668dbff2caa4bb', '0010-zig-glm-tpn-roce-probe.patch': '3b50eb61406891aa93ee0dda45f1b522ed5acb912eaa9005b98514133cf061b9', '0011-zig-glm-greedy-draft-probs.patch': '82a65c4b5fdeed13c5ca183c760a72ca0fe5c80de736195ad3340d02734c672b', '0012-zig-glm-cache-reserve.patch': 'ff94367831b8c401ea35fdad3ebe2141da9646242a2ecb22dba576c6f71e3432', '0013-zig-glm-round-batching.patch': '7be776fa76a73d9159d867f4e75f62653a35c749925c18cf94c57311e1eb8f96'}
ORIGINAL_COMMAND = ['docker', 'build', '--progress=plain', '-t', 'spark/glm-tp4:zig-a6c7e08-baseline', '-f', '/home/spark/lab/glm-tp4-ring-20261022/build-context/Dockerfile', '--build-arg', 'BUILD_IMAGE=nvidia/cuda@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6', '--build-arg', 'RUN_IMAGE=nvidia/cuda@sha256:6a0e31b59e70890446f2c17356d6efc0d54260090a8c63d6ca7c2ad049db95d2', '--build-arg', 'TF_REPO=https://github.com/ashhart/TensorFold.git', '--build-arg', 'TF_REF=e7c523ebfa16b8094cf9e35e3773a7d25befcc5b', '--build-arg', 'ZIG_VERSION=0.17.0', '--build-arg', 'ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8', '--build-arg', 'NCCL_VERSION=2.30.7', '--build-arg', 'AOT_SHA256=29ad991945a30f3398ae79ba781cb975f499b752a5b5dbc473d1c152e6bdb681', '--build-arg', 'PATCHES_HASH=ccee6589d40e82d97236a7feaaf7fa6498c078e61ba638998b9009a7926b0d7b', '/home/spark/lab/glm-tp4-ring-20261022/build-context']
ORIGINAL_DOCKERFILE_SHA256 = '98a37bd2e97d34404a4108e204b69fd5887b0c4f6e51e14e5f5344eadeb7d78c'
INPUT_SHA256 = {'expected-fatbins-be94e91.json': '704744c80e88afd6554750f4b770d3df3b1b1a9c5c1ecc3268aa77c25a1f8542', 'native-source-blobs-be94e91.json': 'd974498a5374a5ede7de6481467d960e55fbdc04a41162ac80e477dd1c61b6c7', 'verify-focused-fatbins-be94e91.py': '1f0d771e25caeff97a5cc8cfbb9c3e750d096167d2580bfd2a3b1582907b53fb', 'verify-focused-source-be94e91.py': 'b10f3621d4d3d534fa047bed550e0f2d3686cfc92d2433e673e0b20dee79f8d0'}
AOT_SHA256 = '29ad991945a30f3398ae79ba781cb975f499b752a5b5dbc473d1c152e6bdb681'
TAG = 'spark/glm-tp4:focused-rdma-be94e91'
HERE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise SystemExit(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclusive(path, data):
    with path.open('xb') as stream:
        stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/spark/lab/glm-tp4-focused-20261022'))
    parser.add_argument('--recipe-commit', required=True, help='Frozen final outer recipe commit archived into context-v2')
    args = parser.parse_args()
    require(re.fullmatch(r'[0-9a-f]{40}', args.recipe_commit), 'Expected full recipe commit')
    root, ctx = args.root, args.root / 'context-v2'
    require(root.is_absolute() and ctx.is_dir() and not ctx.is_symlink(), 'Expected existing context-v2 directory')
    for name, expected in INPUT_SHA256.items():
        require(sha(HERE / name) == expected, 'Build input hash mismatch: ' + name)
    dockerfile = ctx / 'zig/Dockerfile'
    require(sha(dockerfile) == ORIGINAL_DOCKERFILE_SHA256, 'Recipe Dockerfile changed; review transformations')
    require(sha(ctx / 'zig/aot/glm-aot-sm121.tar.xz') == AOT_SHA256, 'Pinned AOT archive differs')
    patches = {p.name: sha(p) for p in sorted((ctx / 'zig/patches').glob('*.patch'))}
    selected = [p for p in patches if p.startswith('0014-')]
    require(len(patches) == 14 and len(selected) == 1 and patches[selected[0]] == PATCH_SHA256,
            'Focused 0014 patch hash or 14-patch inventory differs')
    require({k:v for k,v in patches.items() if k != selected[0]} == FIRST_THIRTEEN,
            'Original thirteen patch hashes differ')
    provenance = json.loads((ctx / 'zig/TRANSPORT_SOURCE.json').read_text())
    # Provenance schemas may grow. Source equality is enforced independently by
    # all 2889 Git blob IDs before compilation; retain the exact provenance hash.
    require(isinstance(provenance, dict), 'Invalid source provenance')
    original = dockerfile.read_text()
    for marker, count in [('-j"$(nproc)"', 2), ('install native &&', 1), ('ARG SM=121\n', 1),
                           ('RUN mkdir -p /opt/tensorfold/share/doc/tensorfold', 1)]:
        require(original.count(marker) == count, 'Unexpected Dockerfile marker: ' + marker)
    preflight = ''.join('COPY ' + name + ' /tmp/' + name + '\n' for name in INPUT_SHA256)
    preflight += 'RUN python3 /tmp/verify-focused-source-be94e91.py && /usr/local/cuda/bin/nvcc --version\n'
    text = original.replace('-j"$(nproc)"', '-j4').replace('install native &&', 'install native glm-transport-probe fatbins &&')
    text = text.replace('ARG SM=121\n', preflight + 'ARG SM=121\n')
    audit = ('RUN cd /tensorfold && /opt/zig/zig run zig/build/check_fatbins.zig -- 121 /opt/tensorfold/fatbin/*.fatbin'
             ' && python3 /tmp/verify-focused-fatbins-be94e91.py'
             ' && test -x /opt/tensorfold/bin/tf-glm-transport-probe\n')
    text = text.replace('RUN mkdir -p /opt/tensorfold/share/doc/tensorfold', audit + 'RUN mkdir -p /opt/tensorfold/share/doc/tensorfold')
    generated = ctx / 'Dockerfile.focused-rdma-be94e91'
    request = root / 'build-request-rdma-be94e91.json'
    receipt_path = root / 'build-receipt-rdma-be94e91.json'
    outputs = [generated, request, receipt_path] + [ctx / name for name in INPUT_SHA256]
    require(not any(p.exists() or p.is_symlink() for p in outputs), 'Existing output refused; preserve prior build attempt')
    exists = subprocess.run(['docker', 'image', 'inspect', TAG], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    require(exists.returncode != 0, 'Image tag already exists; do not overwrite prior artifact')
    exclusive(generated, text.encode())
    for name in INPUT_SHA256:
        exclusive(ctx / name, (HERE / name).read_bytes())
    command = list(ORIGINAL_COMMAND)
    command[command.index('-t') + 1] = TAG
    command[command.index('-f') + 1] = str(generated)
    command[-1] = str(ctx)
    combined = hashlib.sha256(''.join(patches.values()).encode()).hexdigest()
    for i, item in enumerate(command):
        if item.startswith('PATCHES_HASH='):
            command[i] = 'PATCHES_HASH=' + combined
    record = {'recipe_commit': args.recipe_commit, 'native_commit': NATIVE_COMMIT,
              'patches': patches, 'source_manifest_sha256': INPUT_SHA256['native-source-blobs-be94e91.json'],
              'source_provenance_sha256': sha(ctx / 'zig/TRANSPORT_SOURCE.json'),
              'aot_sha256': AOT_SHA256, 'dockerfile_sha256': sha(generated),
              'build_script_sha256': sha(Path(__file__).resolve()), 'inputs_sha256': INPUT_SHA256,
              'command': command, 'build_jobs': 4, 'no_gpu_execution': True,
              'transformations': ['jobs4', 'real native+probe+56 NVCC fatbins in one build',
                  'exact 2889 source blobs before compile; clean source and real verbs required',
                  'exact 56 fatbin names and nonempty bytes; SM121 SASS check'],
              'proof_limits': ['Build only; no GPU or switched hardware qualification',
                  'Stock image NCCL wheel does not replace the separately patched fleet NCCL mount']}
    exclusive(request, (json.dumps(record, indent=2) + '\n').encode())
    subprocess.run(command, check=True)
    image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', TAG]))[0]
    record.update(image_id=image['Id'], image_size=image['Size'],
        image_fingerprint=hashlib.sha256(json.dumps({'rootfs': image['RootFS'], 'config': image['Config']},
            sort_keys=True, separators=(',', ':')).encode()).hexdigest(), passed=True)
    binary_rows = subprocess.check_output(['docker','run','--rm','--runtime','runc','--network','none','--read-only','--user','1000:1000','--cap-drop','ALL','--security-opt','no-new-privileges','--entrypoint','/usr/bin/sha256sum',image['Id'],'/opt/tensorfold/native/bin/tensorfold-native','/opt/tensorfold/bin/tf-glm-transport-probe'],text=True,timeout=120).splitlines()
    binary_hashes = {row.split(None,1)[1]:row.split(None,1)[0] for row in binary_rows}
    record.update(engine=binary_hashes['/opt/tensorfold/native/bin/tensorfold-native'],probe=binary_hashes['/opt/tensorfold/bin/tf-glm-transport-probe'])
    require(all(re.fullmatch(r'[0-9a-f]{64}',record[k]) for k in ('engine','probe')), 'Invalid built binary digest')
    exclusive(receipt_path, (json.dumps(record, indent=2) + '\n').encode())
    print('FOCUSED_RDMA_IMAGE_READY', image['Id'], flush=True)


if __name__ == '__main__':
    main()
