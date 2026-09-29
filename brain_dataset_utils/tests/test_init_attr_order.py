"""Guard: no __init__ reads a self.<attr> before assigning it.

Caught a real bug -- the Phase 3 deep-gray block was inserted above the
`self.lambda_mask_weight` assignment it depends on, so every config with
lambda_deepgray_weight > 1.0 died at model construction with
`AttributeError: 'BrownianBridgeModel' object has no attribute
'lambda_mask_weight'`. Python cannot catch this statically, the guarded
branch meant the control arm ran fine, and it only surfaced 30 seconds into
a queued GPU job.

Only attributes that ARE assigned somewhere in the same __init__ are
considered. Anything set by a parent class or elsewhere is out of scope here.
"""
import ast
import os
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

TARGETS = [
    'model/BrownianBridge/BrownianBridgeModel.py',
    'runners/BaseRunner.py',
    'datasets/base.py',
    'datasets/custom.py',
]


def check_init(fn, class_name, path):
    """Return a list of (attr, read_line, assign_line) violations."""
    assigned = {}   # attr -> first assignment line
    violations = []

    class Visitor(ast.NodeVisitor):
        def visit_Attribute(self, node):
            # Recurse first so `self.a = self.b` sees the read on the RHS.
            self.generic_visit(node)
            if (isinstance(node.value, ast.Name) and node.value.id == 'self'
                    and isinstance(node.ctx, ast.Load)):
                reads.append((node.attr, node.lineno))

    # Walk top-level statements in source order, recording reads then writes.
    for stmt in ast.walk(fn):
        pass

    reads = []
    Visitor().visit(fn)

    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in tgts:
                if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                        and t.value.id == 'self'):
                    if t.attr not in assigned or node.lineno < assigned[t.attr]:
                        assigned[t.attr] = node.lineno

    for attr, line in reads:
        if attr in assigned and line < assigned[attr]:
            violations.append((attr, line, assigned[attr]))
    return violations


def main():
    total = 0
    for rel in TARGETS:
        path = os.path.join(REPO, rel)
        tree = ast.parse(open(path, encoding='utf-8').read())
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body
                       if isinstance(n, ast.FunctionDef) and n.name == '__init__']:
                for attr, read_line, assign_line in check_init(fn, cls.name, path):
                    print(f"  FAIL {rel}:{read_line} {cls.name}.__init__ reads "
                          f"self.{attr} but assigns it at line {assign_line}")
                    total += 1
        print(f"  checked {rel}")
    if total:
        print(f"{total} use-before-assign violation(s)")
        sys.exit(1)
    print("no __init__ reads a self attribute before assigning it")


if __name__ == '__main__':
    main()
