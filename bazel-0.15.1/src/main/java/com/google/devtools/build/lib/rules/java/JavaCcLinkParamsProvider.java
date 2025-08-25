// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.java;

import com.google.common.base.Function;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.AbstractCcLinkParamsStore;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** A target that provides C++ libraries to be linked into Java targets. */
@Immutable
@AutoCodec
public final class JavaCcLinkParamsProvider implements TransitiveInfoProvider {
  private final CcLinkParamsStore store;

  public JavaCcLinkParamsProvider(AbstractCcLinkParamsStore store) {
    this(new CcLinkParamsStore(store));
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  JavaCcLinkParamsProvider(CcLinkParamsStore store) {
    this.store = store;
  }

  public AbstractCcLinkParamsStore getLinkParams() {
    return store;
  }

  public static final Function<TransitiveInfoCollection, AbstractCcLinkParamsStore> TO_LINK_PARAMS =
      input -> {
        JavaCcLinkParamsProvider provider = input.getProvider(JavaCcLinkParamsProvider.class);
        return provider == null ? null : provider.getLinkParams();
      };
}
