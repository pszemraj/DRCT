import os
import subprocess


def get_git_hash():
    def _minimal_ext_cmd(cmd):
        env = {}
        for k in ["SYSTEMROOT", "PATH", "HOME"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        sha = out.strip().decode("ascii")
    except OSError:
        sha = "unknown"

    return sha

def get_hash():
    if os.path.exists(".git"):
        sha = get_git_hash()[:7]
    else:
        sha = "unknown"

    return sha

# Read version from VERSION file
with open("VERSION", "r") as f:
    __version__ = f.read().strip()

__gitsha__ = get_hash()

# Create version_info tuple
version_info = tuple(int(x) if x.isdigit() else x for x in __version__.split("."))
