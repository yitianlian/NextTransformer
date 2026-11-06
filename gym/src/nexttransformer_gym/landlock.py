"""Landlock confinement for native research plugin workers.

Run as ``python -m nexttransformer_gym.landlock POLICY -- COMMAND...``. The launcher
creates a private temporary directory and forks. The child sets no_new_privs,
restricts itself with Landlock and, unless the policy allows Unix sockets, a
seccomp filter, then executes COMMAND, which inherits the restrictions; any
failure stops before COMMAND runs. The parent removes the directory after the
child exits and returns the child's exit status.
"""

import ctypes
import errno
import json
import os
import platform
import signal
import sys
import tempfile
import traceback

from .common import require

# Landlock system call numbers on x86_64 and asm-generic architectures (arm64).
CREATE_RULESET, ADD_RULE, RESTRICT_SELF = 444, 445, 446
CREATE_RULESET_VERSION = 1
RULE_PATH_BENEATH = 1
PR_SET_PDEATHSIG, PR_SET_SECCOMP, PR_SET_NO_NEW_PRIVS = 1, 22, 38
SECCOMP_MODE_FILTER = 2
# Audit architecture and socket(2) number for the seccomp filter.
SECCOMP_ARCH = {"x86_64": (0xC000003E, 41), "aarch64": (0xC00000B7, 198)}

EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
REFER, TRUNCATE = 1 << 13, 1 << 14
FILE_RIGHTS = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE
READ_RIGHTS = EXECUTE | READ_FILE | READ_DIR


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(SockFilter))]


libc = ctypes.CDLL(None, use_errno=True)


def check(result):
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def syscall(*args):
    return check(libc.syscall(*args))


def abi_version():
    try:
        return syscall(CREATE_RULESET, None, 0, CREATE_RULESET_VERSION)
    except OSError:
        return 0


def require_available():
    require(
        abi_version() >= 1 and platform.machine() in SECCOMP_ARCH,
        "resource_unavailable",
        "The landlock isolation backend requires an x86_64 or aarch64 kernel "
        "with Landlock enabled.",
    )


def handled_rights(abi):
    rights = (1 << 13) - 1  # every ABI 1 filesystem right
    if abi >= 2:
        rights |= REFER
    if abi >= 3:
        rights |= TRUNCATE
    return rights


def restrict(read, listed, write):
    """Deny handled filesystem rights outside the given paths.

    Missing paths are skipped. This also sets no_new_privs.
    """
    abi = abi_version()
    if abi < 1:
        raise OSError("Landlock is unavailable.")
    handled = handled_rights(abi)
    attr = RulesetAttr(handled)
    ruleset = syscall(CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    try:
        rules = [(path, READ_RIGHTS) for path in read]
        rules += [(path, READ_DIR) for path in listed]
        rules += [(path, handled) for path in write]
        for path, rights in rules:
            if not os.path.exists(path):
                continue
            rights &= handled if os.path.isdir(path) else handled & FILE_RIGHTS
            if not rights:
                continue
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathBeneathAttr(rights, fd)
                syscall(ADD_RULE, ruleset, RULE_PATH_BENEATH, ctypes.byref(rule), 0)
            finally:
                os.close(fd)
        check(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0))
        syscall(RESTRICT_SELF, ruleset, 0)
    finally:
        os.close(ruleset)


def deny_unix_sockets():
    """Refuse new AF_UNIX sockets; Landlock does not cover Unix socket connects."""
    arch, socket_nr = SECCOMP_ARCH[platform.machine()]
    load, jeq, jge, ret = 0x20, 0x15, 0x35, 0x06
    program = [
        (load, 0, 0, 4),  # architecture
        (jeq, 1, 0, arch),
        (ret, 0, 0, 0x80000000),  # kill the process
        (load, 0, 0, 0),  # system call number
        (jge, 3, 0, 0x40000000),  # x32 system calls
        (jeq, 0, 3, socket_nr),
        (load, 0, 0, 16),  # socket(2) domain
        (jeq, 0, 1, 1),  # AF_UNIX
        (ret, 0, 0, 0x00050000 | errno.EACCES),
        (ret, 0, 0, 0x7FFF0000),  # allow
    ]
    filters = (SockFilter * len(program))(*(SockFilter(*f) for f in program))
    prog = SockFprog(len(program), filters)
    check(libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(prog), 0, 0))


def run_confined(policy, command, home, parent):
    try:
        check(libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0))
        if os.getppid() != parent:
            raise OSError("The launcher exited before the worker started.")
        os.environ.update(
            HOME=home, TMPDIR=home, XDG_CACHE_HOME=os.path.join(home, "cache")
        )
        os.chdir(home)
        restrict(policy["read"], policy["list"], [*policy["write"], home])
        if not policy["unix_sockets"]:
            deny_unix_sockets()
        os.execv(command[0], command)
    except BaseException:
        traceback.print_exc()
    finally:
        os._exit(127)


def main(argv):
    if len(argv) < 3 or argv[1] != "--":
        raise SystemExit(
            "usage: python -m nexttransformer_gym.landlock POLICY -- COMMAND..."
        )
    policy, command = json.loads(argv[0]), argv[2:]
    with tempfile.TemporaryDirectory(prefix="ntgym-worker-", dir=policy["tmp"]) as home:
        parent = os.getpid()
        child = os.fork()
        if child == 0:
            run_confined(policy, command, home, parent)
        _, status = os.waitpid(child, 0)
    code = os.waitstatus_to_exitcode(status)
    sys.exit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main(sys.argv[1:])
