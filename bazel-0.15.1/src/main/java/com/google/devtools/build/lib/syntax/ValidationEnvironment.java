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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/** A class for doing static checks on files, before evaluating them. */
public final class ValidationEnvironment extends SyntaxTreeVisitor {

  private static class Block {
    private final Set<String> variables = new HashSet<>();
    private final Set<String> readOnlyVariables = new HashSet<>();
    @Nullable private final Block parent;

    Block(@Nullable Block parent) {
      this.parent = parent;
    }
  }

  /**
   * We use an unchecked exception around EvalException because the SyntaxTreeVisitor doesn't let
   * visit methods throw checked exceptions. We might change that later.
   */
  private static class ValidationException extends RuntimeException {
    EvalException exception;

    ValidationException(EvalException e) {
      exception = e;
    }

    ValidationException(Location location, String message, String url) {
      exception = new EvalException(location, message, url);
    }

    ValidationException(Location location, String message) {
      exception = new EvalException(location, message);
    }
  }

  private final SkylarkSemantics semantics;
  private Block block;
  private int loopCount;

  /** Create a ValidationEnvironment for a given global Environment. */
  ValidationEnvironment(Environment env) {
    Preconditions.checkArgument(env.isGlobal());
    block = new Block(null);
    Set<String> builtinVariables = env.getVariableNames();
    block.variables.addAll(builtinVariables);
    block.readOnlyVariables.addAll(builtinVariables);
    semantics = env.getSemantics();
  }

  @Override
  public void visit(LoadStatement node) {
    for (Identifier symbol : node.getSymbols()) {
      declare(symbol.getName(), node.getLocation());
    }
  }

  @Override
  public void visit(Identifier node) {
    if (!hasSymbolInEnvironment(node.getName())) {
      throw new ValidationException(node.createInvalidIdentifierException(getAllSymbols()));
    }
  }

  private void validateLValue(Location loc, Expression expr) {
    if (expr instanceof Identifier) {
      declare(((Identifier) expr).getName(), loc);
    } else if (expr instanceof IndexExpression) {
      visit(expr);
    } else if (expr instanceof ListLiteral) {
      for (Expression e : ((ListLiteral) expr).getElements()) {
        validateLValue(loc, e);
      }
    } else {
      throw new ValidationException(loc, "cannot assign to '" + expr + "'");
    }
  }

  @Override
  public void visit(LValue node) {
    validateLValue(node.getLocation(), node.getExpression());
  }

  @Override
  public void visit(ReturnStatement node) {
    if (isTopLevel()) {
      throw new ValidationException(
          node.getLocation(), "return statements must be inside a function");
    }
    super.visit(node);
  }

  @Override
  public void visit(ForStatement node) {
    loopCount++;
    super.visit(node);
    Preconditions.checkState(loopCount > 0);
    loopCount--;
  }

  @Override
  public void visit(FlowStatement node) {
    if (loopCount <= 0) {
      throw new ValidationException(
          node.getLocation(), node.getKind().getName() + " statement must be inside a for loop");
    }
    super.visit(node);
  }

  @Override
  public void visit(DotExpression node) {
    visit(node.getObject());
    // Do not visit the field.
  }

  @Override
  public void visit(AbstractComprehension node) {
    openBlock();
    super.visit(node);
    closeBlock();
  }

  @Override
  public void visit(FunctionDefStatement node) {
    for (Parameter<Expression, Expression> param : node.getParameters()) {
      if (param.isOptional()) {
        visit(param.getDefaultValue());
      }
    }
    openBlock();
    for (Parameter<Expression, Expression> param : node.getParameters()) {
      if (param.hasName()) {
        declare(param.getName(), param.getLocation());
      }
    }
    visitAll(node.getStatements());
    closeBlock();
  }

  @Override
  public void visit(IfStatement node) {
    if (isTopLevel()) {
      throw new ValidationException(
          node.getLocation(),
          "if statements are not allowed at the top level. You may move it inside a function "
          + "or use an if expression (x if condition else y).");
    }
    super.visit(node);
  }

  @Override
  public void visit(AugmentedAssignmentStatement node) {
    if (node.getLValue().getExpression() instanceof ListLiteral) {
      throw new ValidationException(
          node.getLocation(), "cannot perform augmented assignment on a list or tuple expression");
    }
    // Other bad cases are handled when visiting the LValue node.
    super.visit(node);
  }

  /** Returns true if the current block is the top level i.e. has no parent. */
  private boolean isTopLevel() {
    return block.parent == null;
  }

  /** Declare a variable and add it to the environment. */
  private void declare(String varname, Location location) {
    if (block.readOnlyVariables.contains(varname)) {
      throw new ValidationException(
          location,
          String.format("Variable %s is read only", varname),
          "https://bazel.build/versions/master/docs/skylark/errors/read-only-variable.html");
    }
    if (isTopLevel()) {  // top-level values are immutable
      block.readOnlyVariables.add(varname);
    }
    block.variables.add(varname);
  }

  /** Returns true if the symbol exists in the validation environment (or a parent). */
  private boolean hasSymbolInEnvironment(String varname) {
    for (Block b = block; b != null; b = b.parent) {
      if (b.variables.contains(varname)) {
        return true;
      }
    }
    return false;
  }

  /** Returns the set of all accessible symbols (both local and global) */
  private Set<String> getAllSymbols() {
    Set<String> all = new HashSet<>();
    for (Block b = block; b != null; b = b.parent) {
      all.addAll(b.variables);
    }
    return all;
  }

  /** Throws ValidationException if a load() appears after another kind of statement. */
  private static void checkLoadAfterStatement(List<Statement> statements) {
    Location firstStatement = null;

    for (Statement statement : statements) {
      // Ignore string literals (e.g. docstrings).
      if (statement instanceof ExpressionStatement
          && ((ExpressionStatement) statement).getExpression() instanceof StringLiteral) {
        continue;
      }

      if (statement instanceof LoadStatement) {
        if (firstStatement == null) {
          continue;
        }
        throw new ValidationException(
            statement.getLocation(),
            "load() statements must be called before any other statement. "
                + "First non-load() statement appears at "
                + firstStatement
                + ". Use --incompatible_bzl_disallow_load_after_statement=false to temporarily "
                + "disable this check.");
      }

      if (firstStatement == null) {
        firstStatement = statement.getLocation();
      }
    }
  }

  /** Validates the AST and runs static checks. */
  private void validateAst(List<Statement> statements) {
    // Check that load() statements are on top.
    if (semantics.incompatibleBzlDisallowLoadAfterStatement()) {
      checkLoadAfterStatement(statements);
    }

    // Add every function in the environment before validating. This is
    // necessary because functions may call other functions defined
    // later in the file.
    for (Statement statement : statements) {
      if (statement instanceof FunctionDefStatement) {
        FunctionDefStatement fct = (FunctionDefStatement) statement;
        declare(fct.getIdentifier().getName(), fct.getLocation());
      }
    }

    this.visitAll(statements);
  }

  public static void validateAst(Environment env, List<Statement> statements) throws EvalException {
    try {
      ValidationEnvironment venv = new ValidationEnvironment(env);
      venv.validateAst(statements);
      // Check that no closeBlock was forgotten.
      Preconditions.checkState(venv.block.parent == null);
    } catch (ValidationException e) {
      throw e.exception;
    }
  }

  public static boolean validateAst(
      Environment env, List<Statement> statements, EventHandler eventHandler) {
    try {
      validateAst(env, statements);
      return true;
    } catch (EvalException e) {
      if (!e.isDueToIncompleteAST()) {
        eventHandler.handle(Event.error(e.getLocation(), e.getMessage()));
      }
      return false;
    }
  }

  /** Open a new lexical block that will contain the future declarations. */
  private void openBlock() {
    block = new Block(block);
  }

  /** Close a lexical block (and lose all declarations it contained). */
  private void closeBlock() {
    block = Preconditions.checkNotNull(block.parent);
  }

  /**
   * Checks that the AST is using the restricted syntax.
   *
   * <p>Restricted syntax is used by Bazel BUILD files. It forbids function definitions, *args, and
   * **kwargs. This creates a better separation between code and data.
   */
  public static boolean checkBuildSyntax(
      List<Statement> statements, final EventHandler eventHandler) {
    // Wrap the boolean inside an array so that the inner class can modify it.
    final boolean[] success = new boolean[] {true};
    // TODO(laurentlb): Merge with the visitor above when possible (i.e. when BUILD files use it).
    SyntaxTreeVisitor checker =
        new SyntaxTreeVisitor() {

          private void error(ASTNode node, String message) {
            eventHandler.handle(Event.error(node.getLocation(), message));
            success[0] = false;
          }

          @Override
          public void visit(FunctionDefStatement node) {
            error(
                node,
                "function definitions are not allowed in BUILD files. You may move the function to "
                    + "a .bzl file and load it.");
          }

          @Override
          public void visit(ForStatement node) {
            error(
                node,
                "for statements are not allowed in BUILD files. You may inline the loop, move it "
                    + "to a function definition (in a .bzl file), or as a last resort use a list "
                    + "comprehension.");
          }

          @Override
          public void visit(IfStatement node) {
            error(
                node,
                "if statements are not allowed in BUILD files. You may move conditional logic to a "
                    + "function definition (in a .bzl file), or for simple cases use an if "
                    + "expression.");
          }

          @Override
          public void visit(FuncallExpression node) {
            for (Argument.Passed arg : node.getArguments()) {
              if (arg.isStarStar()) {
                error(
                    node,
                    "**kwargs arguments are not allowed in BUILD files. Pass the arguments in "
                        + "explicitly.");
              } else if (arg.isStar()) {
                error(
                    node,
                    "*args arguments are not allowed in BUILD files. Pass the arguments in "
                        + "explicitly.");
              }
            }
          }
        };
    checker.visitAll(statements);
    return success[0];
  }
}
