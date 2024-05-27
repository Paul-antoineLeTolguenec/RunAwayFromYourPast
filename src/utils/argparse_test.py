import argparse
import os
from dataclasses import dataclass, fields


def parse_args(cls):
    parser = argparse.ArgumentParser(description="Your script description")
    for field in fields(cls):
        field_type = field.type
        default_value = field.default
        if field_type == bool:
            # Special handling for boolean values
            parser.add_argument(f"--{field.name}", type=lambda x: (str(x).lower() == 'true'), default=default_value, help=field.metadata.get("help", ""))
        else:
            parser.add_argument(f"--{field.name}", type=field_type, default=default_value, help=field.metadata.get("help", ""))
    args = parser.parse_args()
    return cls(**vars(args))

if __name__ == "__main__":
    @dataclass
    class Args:
        exp_name: str = os.path.basename(__file__)[: -len(".py")]
        seed: int = 0
        torch_deterministic: bool = True

    # test 
    args = parse_args(Args)
