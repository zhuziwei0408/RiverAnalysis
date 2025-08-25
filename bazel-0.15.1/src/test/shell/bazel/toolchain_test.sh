#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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
#
# Test the providers and rules related to toolchains.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  create_new_workspace

  # Create shared report rule for printing toolchain info.
  mkdir report
  touch report/BUILD
  cat >>report/report.bzl <<EOF
def _report_impl(ctx):
  toolchain = ctx.attr.toolchain[platform_common.ToolchainInfo]
  for field in ctx.attr.fields:
    value = getattr(toolchain, field)
    if type(value) == 'Target':
      value = value.label
    print('%s = "%s"' % (field, value))

report_toolchain = rule(
    implementation = _report_impl,
    attrs = {
        'fields': attr.string_list(),
        'toolchain': attr.label(providers = [platform_common.ToolchainInfo]),
    }
)
EOF
}

function write_test_toolchain() {
  toolchain_name=${1:-test_toolchain}
  mkdir -p toolchain
  cat >> toolchain/toolchain_${toolchain_name}.bzl <<EOF
def _impl(ctx):
  toolchain = platform_common.ToolchainInfo(
      extra_label = ctx.attr.extra_label,
      extra_str = ctx.attr.extra_str)
  return [toolchain]

${toolchain_name} = rule(
    implementation = _impl,
    attrs = {
        'extra_label': attr.label(),
        'extra_str': attr.string(),
    }
)
EOF

  cat >> toolchain/BUILD <<EOF
toolchain_type(name = '${toolchain_name}')
EOF
}

function write_test_rule() {
  rule_name=${1:-use_toolchain}
  toolchain_name=${2:-test_toolchain}
  mkdir -p toolchain
  cat >> toolchain/rule_${rule_name}.bzl <<EOF
def _impl(ctx):
  toolchain = ctx.toolchains['//toolchain:${toolchain_name}']
  message = ctx.attr.message
  print(
      'Using toolchain: rule message: "%s", toolchain extra_str: "%s"' %
         (message, toolchain.extra_str))
  return []

${rule_name} = rule(
    implementation = _impl,
    attrs = {
        'message': attr.string(),
    },
    toolchains = ['//toolchain:${toolchain_name}'],
)
EOF
}

function write_test_aspect() {
  aspect_name=${1:-use_toolchain}
  toolchain_name=${2:-test_toolchain}
  mkdir -p toolchain
  cat >> toolchain/aspect_${aspect_name}.bzl <<EOF
def _impl(target, ctx):
  toolchain = ctx.toolchains['//toolchain:${toolchain_name}']
  message = ctx.rule.attr.message
  print(
      'Using toolchain in aspect: rule message: "%s", toolchain extra_str: "%s"' %
          (message, toolchain.extra_str))
  return []

${aspect_name} = aspect(
    implementation = _impl,
    attrs = {},
    toolchains = ['//toolchain:${toolchain_name}'],
)
EOF
}

function write_register_toolchain() {
  toolchain_name=${1:-test_toolchain}
  cat >> WORKSPACE <<EOF
register_toolchains('//:${toolchain_name}_1')
EOF

  cat >> BUILD <<EOF
load('//toolchain:toolchain_${toolchain_name}.bzl', '${toolchain_name}')

# Define the toolchain.
filegroup(name = 'dep_rule_${toolchain_name}')
${toolchain_name}(
    name = '${toolchain_name}_impl_1',
    extra_label = ':dep_rule_${toolchain_name}',
    extra_str = 'foo from ${toolchain_name}',
    visibility = ['//visibility:public'])

# Declare the toolchain.
toolchain(
    name = '${toolchain_name}_1',
    toolchain_type = '//toolchain:${toolchain_name}',
    exec_compatible_with = [],
    target_compatible_with = [],
    toolchain = ':${toolchain_name}_impl_1',
    visibility = ['//visibility:public'])
EOF
}

function test_toolchain_provider() {
  write_test_toolchain

  cat >> BUILD <<EOF
load('//toolchain:toolchain_test_toolchain.bzl', 'test_toolchain')
load('//report:report.bzl', 'report_toolchain')

filegroup(name = 'dep_rule')
test_toolchain(
    name = 'linux_toolchain',
    extra_label = ':dep_rule',
    extra_str = 'bar',
)
report_toolchain(
  name = 'report',
  fields = ['extra_label', 'extra_str'],
  toolchain = ':linux_toolchain',
)
EOF

  bazel build //:report &> $TEST_log || fail "Build failed"
  expect_log 'extra_label = "//:dep_rule"'
  expect_log 'extra_str = "bar"'
}

function test_toolchain_use_in_rule {
  write_test_toolchain
  write_test_rule
  write_register_toolchain

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain extra_str: "foo from test_toolchain"'
}

function test_toolchain_use_in_rule_missing {
  write_test_toolchain
  write_test_rule
  #rite_register_toolchain
  # Do not register test_toolchain to trigger the error.

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log 'While resolving toolchains for target //demo:use: no matching toolchains found for types //toolchain:test_toolchain'
}

function test_multiple_toolchain_use_in_rule {
  write_test_toolchain test_toolchain_1
  write_test_toolchain test_toolchain_2

  write_register_toolchain test_toolchain_1
  write_register_toolchain test_toolchain_2

  # The rule uses two separate toolchains.
  mkdir -p toolchain
  cat >> toolchain/rule_use_toolchains.bzl <<EOF
def _impl(ctx):
  toolchain_1 = ctx.toolchains['//toolchain:test_toolchain_1']
  toolchain_2 = ctx.toolchains['//toolchain:test_toolchain_2']
  message = ctx.attr.message
  print(
      'Using toolchain: rule message: "%s", toolchain 1 extra_str: "%s", toolchain 2 extra_str: "%s"' %
         (message, toolchain_1.extra_str, toolchain_2.extra_str))
  return []

use_toolchains = rule(
    implementation = _impl,
    attrs = {
        'message': attr.string(),
    },
    toolchains = [
        '//toolchain:test_toolchain_1',
        '//toolchain:test_toolchain_2',
    ],
)
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchains.bzl', 'use_toolchains')
# Use the toolchain.
use_toolchains(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain 1 extra_str: "foo from test_toolchain_1", toolchain 2 extra_str: "foo from test_toolchain_2"'
}

function test_multiple_toolchain_use_in_rule_one_missing {
  write_test_toolchain test_toolchain_1
  write_test_toolchain test_toolchain_2

  write_register_toolchain test_toolchain_1
  # Do not register test_toolchain_2 to cause the error,

  # The rule uses two separate toolchains.
  mkdir -p toolchain
  cat >> toolchain/rule_use_toolchains.bzl <<EOF
def _impl(ctx):
  toolchain_1 = ctx.toolchains['//toolchain:test_toolchain_1']
  toolchain_2 = ctx.toolchains['//toolchain:test_toolchain_2']
  message = ctx.attr.message
  print(
      'Using toolchain: rule message: "%s", toolchain 1 extra_str: "%s", toolchain 2 extra_str: "%s"' %
         (message, toolchain_1.extra_str, toolchain_2.extra_str))
  return []

use_toolchains = rule(
    implementation = _impl,
    attrs = {
        'message': attr.string(),
    },
    toolchains = [
        '//toolchain:test_toolchain_1',
        '//toolchain:test_toolchain_2',
    ],
)
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchains.bzl', 'use_toolchains')
# Use the toolchain.
use_toolchains(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log 'While resolving toolchains for target //demo:use: no matching toolchains found for types //toolchain:test_toolchain_2'
}

function test_toolchain_use_in_rule_non_required_toolchain {
  write_test_toolchain
  write_register_toolchain

  # The rule argument toolchains requires one toolchain, but the implementation requests a different
  # one.
  mkdir -p toolchain
  cat >> toolchain/rule_use_toolchain.bzl <<EOF
def _impl(ctx):
  toolchain = ctx.toolchains['//toolchain:wrong_toolchain']
  message = ctx.attr.message
  print(
      'Using toolchain: rule message: "%s", toolchain extra_str: "%s"' %
         (message, toolchain.extra_str))
  return []

use_toolchain = rule(
    implementation = _impl,
    attrs = {
        'message': attr.string(),
    },
    toolchains = ['//toolchain:test_toolchain'],
)
EOF

  # Trigger the wrong toolchain.
  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log 'In use_toolchain rule //demo:use, toolchain type //toolchain:wrong_toolchain was requested but only types \[//toolchain:test_toolchain\] are configured'
}

function test_toolchain_debug_messages {
  write_test_toolchain
  write_test_rule
  write_register_toolchain

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build \
    --toolchain_resolution_debug \
    //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'ToolchainResolution: Looking for toolchain of type //toolchain:test_toolchain'
  expect_log 'ToolchainResolution:   For toolchain type //toolchain:test_toolchain, possible execution platforms and toolchains: {@bazel_tools//platforms:host_platform -> //:test_toolchain_impl_1}'
  expect_log 'ToolchainUtil: Selected execution platform @bazel_tools//platforms:host_platform, type //toolchain:test_toolchain -> toolchain //:test_toolchain_impl_1'
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain extra_str: "foo from test_toolchain"'
}

function test_toolchain_use_in_aspect {
  write_test_toolchain
  write_test_aspect
  write_register_toolchain

  mkdir -p demo
  cat >> demo/demo.bzl <<EOF
def _impl(ctx):
  return []

demo = rule(
    implementation = _impl,
    attrs = {
        'message': attr.string(),
    }
)
EOF
  cat >> demo/BUILD <<EOF
load(':demo.bzl', 'demo')
demo(
    name = 'demo',
    message = 'bar from demo')
EOF

  bazel build \
    --aspects //toolchain:aspect_use_toolchain.bzl%use_toolchain \
    //demo:demo &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain in aspect: rule message: "bar from demo", toolchain extra_str: "foo from test_toolchain"'
}

function test_toolchain_use_in_aspect_non_required_toolchain {
  write_test_toolchain
  write_register_toolchain

  # The aspect argument toolchains requires one toolchain, but the implementation requests a
  # different one.
  mkdir -p toolchain
  cat >> toolchain/aspect_use_toolchain.bzl <<EOF
def _impl(target, ctx):
  toolchain = ctx.toolchains['//toolchain:wrong_toolchain']
  message = ctx.rule.attr.message
  print(
      'Using toolchain in aspect: rule message: "%s", toolchain extra_str: "%s"' %
          (message, toolchain.extra_str))
  return []

use_toolchain = aspect(
    implementation = _impl,
    attrs = {},
    toolchains = ['//toolchain:test_toolchain'],
)
EOF

  # Trigger the wrong toolchain.
  mkdir -p demo
  cat >> demo/demo.bzl <<EOF
def _impl(ctx):
  return []

demo = rule(
    implementation = _impl,
    attrs = {
        'message': attr.string(),
    }
)
EOF
  cat >> demo/BUILD <<EOF
load(':demo.bzl', 'demo')
demo(
    name = 'demo',
    message = 'bar from demo')
EOF

  bazel build \
    --aspects //toolchain:aspect_use_toolchain.bzl%use_toolchain \
    //demo:demo &> $TEST_log && fail "Build failure expected"
  expect_log 'In aspect //toolchain:aspect_use_toolchain.bzl%use_toolchain applied to demo rule //demo:demo, toolchain type //toolchain:wrong_toolchain was requested but only types \[//toolchain:test_toolchain\] are configured'
}

function test_toolchain_constraints() {
  write_test_toolchain
  write_test_rule

  cat >> WORKSPACE <<EOF
register_toolchains('//:toolchain_1')
register_toolchains('//:toolchain_2')
EOF

  cat >> BUILD <<EOF
load('//toolchain:toolchain_test_toolchain.bzl', 'test_toolchain')

# Define constraints.
constraint_setting(name = 'setting')
constraint_value(name = 'value1', constraint_setting = ':setting')
constraint_value(name = 'value2', constraint_setting = ':setting')

platform(
    name = 'platform1',
    constraint_values = [':value1'],
    visibility = ['//visibility:public'])
platform(
    name = 'platform2',
    constraint_values = [':value2'],
    visibility = ['//visibility:public'])

# Define the toolchain.
filegroup(name = 'dep_rule')
test_toolchain(
    name = 'toolchain_impl_1',
    extra_label = ':dep_rule',
    extra_str = 'foo from 1',
    visibility = ['//visibility:public'])
test_toolchain(
    name = 'toolchain_impl_2',
    extra_label = ':dep_rule',
    extra_str = 'foo from 2',
    visibility = ['//visibility:public'])

# Declare the toolchain.
toolchain(
    name = 'toolchain_1',
    toolchain_type = '//toolchain:test_toolchain',
    exec_compatible_with = [':value1'],
    target_compatible_with = [':value2'],
    toolchain = ':toolchain_impl_1')
toolchain(
    name = 'toolchain_2',
    toolchain_type = '//toolchain:test_toolchain',
    exec_compatible_with = [':value2'],
    target_compatible_with = [':value1'],
    toolchain = ':toolchain_impl_2')
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  # This should use toolchain_1.
  bazel build \
    --host_platform=//:platform1 \
    --platforms=//:platform2 \
    //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain extra_str: "foo from 1"'

  # This should use toolchain_2.
  bazel build \
    --host_platform=//:platform2 \
    --platforms=//:platform1 \
    //demo:use &> $TEST_log || fail "Build failed"
  expect_log 'Using toolchain: rule message: "this is the rule", toolchain extra_str: "foo from 2"'

  # This should not match any toolchains.
  bazel build \
    --host_platform=//:platform1 \
    --platforms=//:platform1 \
    //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log 'While resolving toolchains for target //demo:use: no matching toolchains found for types //toolchain:test_toolchain'
  expect_not_log 'Using toolchain: rule message:'
}

function test_register_toolchain_error_invalid_label() {
  write_test_toolchain
  write_test_rule
  write_register_toolchain

  cat >> WORKSPACE <<EOF
register_toolchains('/:invalid:label:syntax')
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log "invalid registered toolchain '/:invalid:label:syntax': not a valid absolute pattern"
}

function test_register_toolchain_error_invalid_target() {
  write_test_toolchain
  write_test_rule
  write_register_toolchain

  cat > WORKSPACE <<EOF
register_toolchains('//demo:not_a_target')
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log "While resolving toolchains for target //demo:use: invalid registered toolchain '//demo:not_a_target': no such target '//demo:not_a_target': target 'not_a_target' not declared in package 'demo'"
}

function test_register_toolchain_error_target_not_a_toolchain() {
  write_test_toolchain
  write_test_rule
  write_register_toolchain

  cat >> WORKSPACE <<EOF
register_toolchains('//demo:invalid')
EOF

  mkdir -p demo
  cat >> demo/out.log<<EOF
INVALID
EOF
  cat >> demo/BUILD <<EOF
filegroup(
    name = "invalid",
    srcs = ["out.log"],
)

load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log "While resolving toolchains for target //demo:use: invalid registered toolchain '//demo:invalid': target does not provide the DeclaredToolchainInfo provider"
}


function test_register_toolchain_error_invalid_pattern() {
  cat >WORKSPACE <<EOF
register_toolchains('//:bad1')
register_toolchains('//:bad2')
EOF

  cat >rules.bzl <<EOF
def _impl(ctx):
  toolchain = ctx.toolchains['//:dummy']
  return []

foo = rule(
  implementation = _impl,
  toolchains = ['//:dummy'],
)
EOF

  cat >BUILD <<EOF
load(":rules.bzl", "foo")
toolchain_type(name = 'dummy')
foo(name = "foo")
EOF

  bazel build //:foo &> $TEST_log && fail "Build failure expected"
  # It's uncertain which error will happen first, so handle either.
  expect_log "While resolving toolchains for target //:foo: invalid registered toolchain '//:bad[12]': no such target"
}


function test_toolchain_error_invalid_target() {
  write_test_toolchain
  write_test_rule

  # Write toolchain with an invalid target.
  mkdir -p invalid
  cat > invalid/BUILD <<EOF
toolchain(
    name = 'invalid_toolchain',
    toolchain_type = '//toolchain:test_toolchain',
    exec_compatible_with = [],
    target_compatible_with = [],
    toolchain = '//toolchain:does_not_exist',
    visibility = ['//visibility:public'])
EOF

  cat > WORKSPACE <<EOF
register_toolchains('//invalid:invalid_toolchain')
EOF

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  bazel build //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log "While resolving toolchains for target //demo:use: no such target '//toolchain:does_not_exist': target 'does_not_exist' not declared in package 'toolchain'"
}


function test_platforms_options_error_invalid_target() {
  write_test_toolchain
  write_test_rule
  write_register_toolchain

  mkdir -p demo
  cat >> demo/BUILD <<EOF
load('//toolchain:rule_use_toolchain.bzl', 'use_toolchain')
# Use the toolchain.
use_toolchain(
    name = 'use',
    message = 'this is the rule')
EOF

  # Write an invalid rule to be the platform.
  mkdir -p platform
  cat >> platform/BUILD <<EOF
filegroup(name = 'not_a_platform')
EOF

  bazel build --platforms=//platform:not_a_platform //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log "While resolving toolchains for target //demo:use: Target //platform:not_a_platform was referenced as a platform, but does not provide PlatformInfo"

  bazel build --host_platform=//platform:not_a_platform //demo:use &> $TEST_log && fail "Build failure expected"
  expect_log "While resolving toolchains for target //demo:use: Target //platform:not_a_platform was referenced as a platform, but does not provide PlatformInfo"
}


run_suite "toolchain tests"
