#!/usr/bin/env python3

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Updates coverage of TensorFlow e2e tests on all backends.

Example usage: python3 update_e2e_coverage.py build-docs
"""

import argparse
import collections
import os
import re
import subprocess

import utils

REFERENCE_BACKEND = 'tf'
# Assumes that tests are expanded for the tf, iree_vmla, iree_llvmjit and
# iree_vulkan backends.
BACKENDS_TO_TITLES = collections.OrderedDict([
    ('tf', 'tensorflow'),
    ('tflite', 'tflite'),
    ('iree_vmla', 'vmla'),
    ('iree_llvmjit', 'llvm-ir'),
    ('iree_vulkan', 'vulkan-spirv'),
])

TEST_SUITES_TO_HEADERS = {
    '//integrations/tensorflow/e2e:e2e_tests':
        'End to end TensorFlow tests',
    '//integrations/tensorflow/e2e:mobile_bert_squad_tests':
        'End to end test of MobileBert on SQuAD',
    '//integrations/tensorflow/e2e/keras:keras_tests':
        'End to end tests written using tf.keras',
    '//integrations/tensorflow/e2e/keras:imagenet_external_tests':
        'End to end tests of tf.keras.applications vision models on Imagenet',
    '//integrations/tensorflow/e2e/slim_vision_models:slim_vision_tests':
        'End to end tests of TensorFlow slim vision models',
}

# Some test suites are generated from a single source. This allows us to point
# to the right test file when generating test URLs.
SINGLE_SOURCE_SUITES = {
    '//integrations/tensorflow/e2e/keras:imagenet_external_tests':
        'vision_model_test',
    '//integrations/tensorflow/e2e/slim_vision_models:slim_vision_tests':
        'slim_vision_model_test',
}

TARGET_EXCLUSION_FILTERS = [
    r'mobilenet_v1_.*',  # Slim vision MobileNetV1.
    r'mobilenet_v2_.*',  # Slim vision MobileNetV2.
    r'amoebanet_a_n18_f448',  # SavedModelV2 not available.
]

# The symbols to show in the table if the operation is supported or not.
SUCCESS_ELEMENT = '<span class="success-table-element">✓</span>'
FAILURE_ELEMENT = '<span class="failure-table-element">✗</span>'

MAIN_URL = 'https://github.com/google/iree/tree/main'
TARGETS_URL = os.path.join(MAIN_URL, 'iree/compiler/Dialect/HAL/Target')

E2E_COVERAGE_DESCRIPTION = f"""# TensorFlow End to End Coverage
There are three backend [targets]({TARGETS_URL}) in IREE:

- vmla
- llvm-ir
- vulkan-spirv

The table shows the supported TensorFlow functions and models on each backend.
It is auto-generated from IREE's test status.

"""


def parse_arguments():
  """Parses command-line options."""
  parser = argparse.ArgumentParser(
      description='Generates Markdown files for op coverage table')
  parser.add_argument('build_dir',
                      metavar='BUILD_PATH',
                      type=str,
                      help='Base build directory.')

  parsed_args = parser.parse_args()
  if not os.path.isdir(parsed_args.build_dir):
    raise parser.error('expected path to a directory')

  return parsed_args


def get_name_and_backend(test_string):
  """Splits a pathless test target into its name and comparison backend."""
  name, backend = test_string.split(f'__{REFERENCE_BACKEND}__')
  return name, backend


def get_suite_metadata(test_suite):
  """Gets all test names, and passing and failing test-backend pairs."""
  passing = utils.get_test_targets(test_suite)
  failing = utils.get_test_targets(f'{test_suite}_failing')

  # Remove bazel path.
  passing = [test.replace(f'{test_suite}_', '') for test in passing]
  failing = [test.replace(f'{test_suite}_failing_', '') for test in failing]

  # Split into (test_name, target_backend).
  passing = [get_name_and_backend(test) for test in passing]
  failing = [get_name_and_backend(test) for test in failing]
  passing_names = [test[0] for test in passing]
  failing_names = [test[0] for test in failing]
  all_names = list(sorted(set(passing_names + failing_names)))
  return all_names, passing, failing


def get_name_element(test_suite, name):
  """Returns a Markdown hyperlink pointing to the test source on GitHub."""
  # Convert `//path/to/tests:test_suite` to `path/to/tests`
  test_path = test_suite.split(':')[0]
  test_path = test_path.replace('//', '')

  if test_suite in SINGLE_SOURCE_SUITES:
    test_name = SINGLE_SOURCE_SUITES[test_suite]
  else:
    test_name = name

  test_url = os.path.join(MAIN_URL, test_path, f'{test_name}.py')
  return f'[{name}]({test_url})'


def generate_table(test_suite):
  """Generates an e2e backend coverage Markdown table."""
  all_names, passing, _ = get_suite_metadata(test_suite)

  # Generate a dictionary mapping test names to their backend coverage.
  table = collections.defaultdict(lambda: [False] * len(BACKENDS_TO_TITLES))
  ordered_backends = list(BACKENDS_TO_TITLES.keys())
  for name, backend in passing:
    table[name][ordered_backends.index(backend)] = True

  # Create a header for the coverage table.
  ordered_backend_titles = list(BACKENDS_TO_TITLES.values())
  first_row = ['target'] + ordered_backend_titles
  second_row = [':-:' for _ in first_row]

  # Generate the coverage table as a 2D array.
  rows = [first_row, second_row]
  for name, backends in sorted(table.items()):
    if any(re.match(pattern, name) for pattern in TARGET_EXCLUSION_FILTERS):
      continue

    row = [get_name_element(test_suite, name)]
    row.extend([
        SUCCESS_ELEMENT if backend else FAILURE_ELEMENT for backend in backends
    ])
    rows.append(row)
  return utils.create_markdown_table(rows)


if __name__ == '__main__':
  args = parse_arguments()

  content = []
  for test_suite, header in TEST_SUITES_TO_HEADERS.items():
    content.append(f'## {header}')
    content.append(generate_table(test_suite))
  content = '\n\n'.join(content) + '\n'  # Trailing newline.

  table_path = os.path.join(args.build_dir, 'doc', 'tf_e2e_coverage.md')
  with open(table_path, 'w', encoding='utf-8') as f:
    f.write(E2E_COVERAGE_DESCRIPTION)
    f.write(content)
