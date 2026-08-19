set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/380_distill_deploy.log") 2>&1
echo "=== deploy distill.py (code only; training starts when teachers exist) ==="
tar xzf tools/myoicl_distill.tar.gz -C .
python -c "import ast; ast.parse(open('myoicl/distill.py').read()); print('AST OK')" || exit 1
git add -A myoicl && git commit -q -m "V4 step 2: teacher-distillation amortizer" 2>&1 | tail -1 || true
echo "deployed"
