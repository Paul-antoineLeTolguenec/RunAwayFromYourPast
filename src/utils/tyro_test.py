import os
from dataclasses import dataclass
import sys

@dataclass
class Args:
    # XP RECORD
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """whether to use torch deterministic"""

def convert_str_to_bool():
    """Convert command line string args to appropriate types."""
    argv = sys.argv
    new_argv = []
    skip = False
    for i, arg in enumerate(argv):
        if skip:
            skip = False
            continue
        if arg.startswith('--'):
            if arg == '--torch_deterministic':
                value = argv[i + 1]
                if value.lower() in ['true', 'false']:
                    new_argv.append(arg)
                    new_argv.append('True' if value.lower() == 'true' else 'False')
                    skip = True
                else:
                    raise ValueError(f"Invalid value for --torch_deterministic: {value}")
            else:
                new_argv.append(arg)
        else:
            new_argv.append(arg)
    sys.argv = new_argv

if __name__ == "__main__":
    import tyro

    convert_str_to_bool()

    args = tyro.cli(Args)
    print(args)
