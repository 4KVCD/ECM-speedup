# long_lines.py <worktree> <path...>: the lines of each file that are not in HEAD's version and are longer than 120
# columns (the style rule for added lines; the per-picture printf lines aside). Run before committing.
import difflib, subprocess, sys

GIT = r'C:\Code\ECM\tools\git\cmd\git.exe'
tree = sys.argv[1]
for path in sys.argv[2:]:
    head = subprocess.run([GIT, '-C', tree, 'show', 'HEAD:' + path], capture_output=True)
    head = head.stdout.decode('utf-8', 'replace')
    new = open(tree + '/' + path, encoding='utf-8', errors='replace').read()
    a, b = head.replace('\r\n', '\n').split('\n'), new.replace('\r\n', '\n').split('\n')
    for line in difflib.unified_diff(a, b, lineterm='', n=0):
        if line.startswith('+') and not line.startswith('+++') and len(line) - 1 > 120 \
           and '[gpu-intra] POC' not in line and 'm.rec.size() - numNodes' not in line:
            print(f'{path}: {len(line) - 1}: {line[1:100]}')
