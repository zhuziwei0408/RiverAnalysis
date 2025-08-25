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
package com.google.devtools.build.lib.syntax;

import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Syntax node for a Parameter in a function (or lambda) definition; it's a subclass of Argument,
 * and contrasts with the class Argument.Passed of arguments in a function call.
 *
 * <p>There are four concrete subclasses of Parameter: Mandatory, Optional, Star, StarStar.
 *
 * <p>See FunctionSignature for how a valid list of Parameter's is organized as a signature, e.g.
 * def foo(mandatory, optional = e1, *args, mandatorynamedonly, optionalnamedonly = e2, **kw): ...
 *
 * <p>V is the class of a defaultValue (Expression at compile-time, Object at runtime),
 * T is the class of a type (Expression at compile-time, SkylarkType at runtime).
 */
public abstract class Parameter<V, T> extends Argument {

  @Nullable protected final String name;
  @Nullable protected final T type;

  private Parameter(@Nullable String name, @Nullable T type) {
    this.name = name;
    this.type = type;
  }

  public boolean isMandatory() {
    return false;
  }

  public boolean isOptional() {
    return false;
  }

  @Override
  public boolean isStar() {
    return false;
  }

  @Override
  public boolean isStarStar() {
    return false;
  }

  @Nullable
  public String getName() {
    return name;
  }

  public boolean hasName() {
    return true;
  }

  @Nullable
  public T getType() {
    return type;
  }

  @Nullable
  public V getDefaultValue() {
    return null;
  }

  /** mandatory parameter (positional or key-only depending on position): Ident */
  @AutoCodec
  public static final class Mandatory<V, T> extends Parameter<V, T> {

    public Mandatory(String name) {
      this(name, null);
    }

    @AutoCodec.Instantiator
    public Mandatory(String name, @Nullable T type) {
      super(name, type);
    }

    @Override
    public boolean isMandatory() {
      return true;
    }

    @Override
    public void prettyPrint(Appendable buffer) throws IOException {
      buffer.append(name);
    }
  }

  /** optional parameter (positional or key-only depending on position): Ident = Value */
  @AutoCodec
  public static final class Optional<V, T> extends Parameter<V, T> {

    public final V defaultValue;

    public Optional(String name, @Nullable V defaultValue) {
      this(name, null, defaultValue);
    }

    @AutoCodec.Instantiator
    public Optional(String name, @Nullable T type, @Nullable V defaultValue) {
      super(name, type);
      this.defaultValue = defaultValue;
    }

    @Override
    @Nullable
    public V getDefaultValue() {
      return defaultValue;
    }

    @Override
    public boolean isOptional() {
      return true;
    }

    @Override
    public void prettyPrint(Appendable buffer) throws IOException {
      buffer.append(name);
      buffer.append('=');
      // This should only ever be used on a parameter representing static information, i.e. with V
      // and T instantiated as Expression.
      ((Expression) defaultValue).prettyPrint(buffer);
    }

    // Keep this as a separate method so that it can be used regardless of what V and T are
    // parameterized with.
    @Override
    public String toString() {
      return name + "=" + defaultValue;
    }
  }

  /** extra positionals parameter (star): *identifier */
  @AutoCodec
  public static final class Star<V, T> extends Parameter<V, T> {

    @AutoCodec.Instantiator
    public Star(@Nullable String name, @Nullable T type) {
      super(name, type);
    }

    public Star(@Nullable String name) {
      this(name, null);
    }

    @Override
    public boolean hasName() {
      return name != null;
    }

    @Override
    public boolean isStar() {
      return true;
    }

    @Override
    public void prettyPrint(Appendable buffer) throws IOException {
      buffer.append('*');
      if (name != null) {
        buffer.append(name);
      }
    }
  }

  /** extra keywords parameter (star_star): **identifier */
  @AutoCodec
  public static final class StarStar<V, T> extends Parameter<V, T> {

    @AutoCodec.Instantiator
    public StarStar(String name, @Nullable T type) {
      super(name, type);
    }

    public StarStar(String name) {
      this(name, null);
    }

    @Override
    public boolean isStarStar() {
      return true;
    }

    @Override
    public void prettyPrint(Appendable buffer) throws IOException {
      buffer.append("**");
      buffer.append(name);
    }
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit((Parameter<Expression, Expression>) this);
  }
}
