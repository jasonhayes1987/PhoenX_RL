# PhoenX integrated-terminal init for Git Bash.
# Used as: bash --init-file scripts/activate.sh -i
# The long option must precede the short ones; bash rejects "-i --init-file".
# Replaces ~/.bashrc for this shell — restore profile/bashrc, never exit, never set -e.

# --init-file skips login profile; without -l we lose /etc/profile PATH entries.
[ -f /etc/profile ] && . /etc/profile
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_record="${_script_dir}/../.phoenx-env"

if [ -f "$_record" ]; then
  # shellcheck disable=SC1090
  . "$_record"
elif [ -n "${PHOENX_ENV_PREFIX:-}" ]; then
  :
else
  echo "No .phoenx-env record found. Run scripts/use-env.ps1 -Prefix <path> or setup.ps1 first."
  unset _script_dir _record
  return 0 2>/dev/null || true
fi

if [ -z "${PHOENX_ENV_PREFIX:-}" ]; then
  echo "No .phoenx-env record found. Run scripts/use-env.ps1 -Prefix <path> or setup.ps1 first."
  unset _script_dir _record
  return 0 2>/dev/null || true
fi

# Stale per-user site-packages can shadow conda's typing_extensions ("Sentinel").
export PYTHONNOUSERSITE=1

if [ -z "${PHOENX_CONDA_ROOT:-}" ]; then
  echo "PHOENX_CONDA_ROOT is unset; cannot locate conda.exe. Re-run use-env.ps1."
  unset _script_dir _record
  return 0 2>/dev/null || true
fi

# Prefer cygpath so we can invoke conda.exe with a POSIX path under Git Bash.
if command -v cygpath >/dev/null 2>&1; then
  conda_root_posix="$(cygpath -u "$PHOENX_CONDA_ROOT")"
else
  conda_root_posix="$PHOENX_CONDA_ROOT"
fi
conda_exe="$conda_root_posix/Scripts/conda.exe"

if [ ! -x "$conda_exe" ]; then
  echo "conda.exe not found or not executable at: $conda_exe"
  unset _script_dir _record conda_root_posix conda_exe
  return 0 2>/dev/null || true
fi

# Quote the prefix: unquoted Windows paths collapse (backslash = escape in bash).
eval "$("$conda_exe" shell.bash hook)"
conda activate "$PHOENX_ENV_PREFIX"
echo "Activated PhoenX env: $PHOENX_ENV_PREFIX"

unset _script_dir _record conda_root_posix conda_exe
