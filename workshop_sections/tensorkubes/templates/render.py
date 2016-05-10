# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from jinja2 import Environment, FileSystemLoader
import yaml
import os


def main(config_name, out):
    env = Environment(
        loader=FileSystemLoader(os.path.dirname(os.path.realpath(__file__))),
        trim_blocks=True,
        lstrip_blocks=True
    )

    with open(config_name) as config_file:
        config = yaml.load(config_file)

    imports = {
        fetch['name']: env.get_template(fetch['path'])
        for fetch in config['imports']
    }

    resources = [
        imports[item['type']].render(properties=item['properties'])
        for item in config['resources']
    ]

    if out is None:
        out = config_name + '.rendered.yaml'

    with open(out, 'w') as out_file:
        out_file.write('---\n'.join(resources))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a Kubernetes config for your Endpoints API')

    # Required
    parser.add_argument(
        '--out',
        help='Location to which the config file should be written',
        default=None
    )
    parser.add_argument(
        '--config',
        help='config file to read configs from'
    )
    parsed = parser.parse_args()
    main(parsed.config, parsed.out)
