// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/IREE/IR/IREEAttributes.h"

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"

// Include TableGen'erated enums
#include "iree/compiler/Dialect/IREE/IR/IREECodeGenEnums.cpp.inc"

// Include TableGen'erated attributes
#include "iree/compiler/Dialect/IREE/IR/IREECodeGenAttributes.cpp.inc"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace CodeGen {

namespace {

template <typename EnumClass>
ParseResult parseEnumKeywordAttr(DialectAsmParser &parser,
                                 StringRef enumClassName,
                                 StringRef &enumValue) {
  StringRef keyword;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&keyword)) return failure();
  if (IREE::CodeGen::symbolizeEnum<EnumClass>(keyword).hasValue()) {
    enumValue = keyword;
    return success();
  }
  return parser.emitError(loc, "invalid ")
         << enumClassName << " attribute value: " << keyword;
}

}  // namespace

Attribute ActionDistributeAttr::parse(DialectAsmParser &parser) {
  ArrayAttr alongDimensions;
  ArrayAttr withWorkgroupSize;

  if (failed(parser.parseLess()) ||
      failed(parser.parseKeyword("along_dimensions")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(alongDimensions)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("with_workgroup_size")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(withWorkgroupSize)) ||
      failed(parser.parseGreater())) {
    return {};
  }

  return ActionDistributeAttr::get(alongDimensions, withWorkgroupSize);
}

void ActionDistributeAttr::print(DialectAsmPrinter &printer) const {
  auto &os = printer.getStream();
  os << getKindName() << "<";
  os << "along_dimensions = " << along_dimensions() << ", ";
  os << "with_workgroup_size = " << with_workgroup_size();
  os << ">";
}

Attribute ActionTileAttr::parse(DialectAsmParser &parser) {
  ArrayAttr alongDimensions;
  ArrayAttr withTileSizes;

  if (failed(parser.parseLess()) ||
      failed(parser.parseKeyword("along_dimensions")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(alongDimensions)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("with_tile_sizes")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(withTileSizes)) ||
      failed(parser.parseGreater())) {
    return {};
  }

  return ActionTileAttr::get(alongDimensions, withTileSizes);
}

void ActionTileAttr::print(DialectAsmPrinter &printer) const {
  auto &os = printer.getStream();
  os << getKindName() << "<";
  os << "along_dimensions = " << along_dimensions() << ", ";
  os << "with_tile_sizes = " << with_tile_sizes();
  os << ">";
}

Attribute ActionTileAndDistributeAttr::parse(DialectAsmParser &parser) {
  ArrayAttr alongDimensions;
  ArrayAttr withTileSizes;
  StringRef toHierarchyLevel;

  if (failed(parser.parseLess()) ||
      failed(parser.parseKeyword("along_dimensions")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(alongDimensions)) ||
      failed(parser.parseComma()) ||
      failed(parser.parseKeyword("with_tile_sizes")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(withTileSizes)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumKeywordAttr<IREE::CodeGen::ComputeHierarchyLevel>(
          parser, "ComputeHierarchyLevel", toHierarchyLevel)) ||
      failed(parser.parseGreater())) {
    return {};
  }

  return ActionTileAndDistributeAttr::get(
      alongDimensions, withTileSizes,
      parser.getBuilder().getStringAttr(toHierarchyLevel));
}

void ActionTileAndDistributeAttr::print(DialectAsmPrinter &printer) const {
  auto &os = printer.getStream();
  os << getKindName() << "<";
  os << "along_dimensions = " << along_dimensions() << ", ";
  os << "with_tile_sizes = " << with_tile_sizes() << ", ";
  os << to_hierarchy_level();
  os << ">";
}

Attribute ActionVectorizeAttr::parse(DialectAsmParser &parser) {
  IntegerAttr withVectorSize;

  if (failed(parser.parseLess()) ||
      failed(parser.parseKeyword("with_vector_size")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(withVectorSize)) ||
      failed(parser.parseGreater())) {
    return {};
  }

  return ActionVectorizeAttr::get(withVectorSize);
}

void ActionVectorizeAttr::print(DialectAsmPrinter &printer) const {
  auto &os = printer.getStream();
  os << getKindName() << "<";
  os << "with_vector_size = " << with_vector_size();
  os << ">";
}

Attribute TypeFilterAttr::parse(DialectAsmParser &parser) {
  auto builder = parser.getBuilder();

  StringRef matchCriterion;
  Type type;

  if (failed(parser.parseLess()) ||
      failed(parseEnumKeywordAttr<IREE::CodeGen::TypeMatchCriterion>(
          parser, "TypeMatchCriterion", matchCriterion)) ||
      failed(parser.parseColon()) || failed(parser.parseType(type)) ||
      failed(parser.parseGreater())) {
    return {};
  }

  return TypeFilterAttr::get(builder.getStringAttr(matchCriterion),
                             TypeAttr::get(type));
}

void TypeFilterAttr::print(DialectAsmPrinter &printer) const {
  auto &os = printer.getStream();
  os << getKindName() << "<" << match_criterion() << ":" << type() << ">";
}

Attribute OpFilterAttr::parse(DialectAsmParser &parser) {
  ArrayAttr onOutputTypes;

  if (failed(parser.parseLess()) ||
      failed(parser.parseKeyword("on_output_types")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseAttribute(onOutputTypes)) ||
      failed(parser.parseGreater())) {
    return {};
  }

  return OpFilterAttr::get(onOutputTypes);
}

void OpFilterAttr::print(DialectAsmPrinter &printer) const {
  auto &os = printer.getStream();
  os << getKindName() << "<";
  os << "on_output_types = " << on_output_types();
  os << ">";
}

Attribute OpChoiceAttr::parse(DialectAsmParser &parser) {
  OpFilterAttr filter;
  ArrayAttr actions;

  if (failed(parser.parseLess()) || failed(parser.parseAttribute(filter)) ||
      failed(parser.parseArrow()) || failed(parser.parseAttribute(actions)) ||
      failed(parser.parseGreater())) {
    return {};
  }

  return OpChoiceAttr::get(filter, actions);
}

void OpChoiceAttr::print(DialectAsmPrinter &printer) const {
  auto &os = printer.getStream();
  os << getKindName() << "<";
  os << filter() << " -> " << actions();
  os << ">";
}

Attribute OpPolicyAttr::parse(DialectAsmParser &parser) {
  auto builder = parser.getBuilder();

  StringAttr op;
  StringRef matchCriterion;
  ArrayAttr choices;

  if (failed(parser.parseLess()) || failed(parser.parseAttribute(op)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumKeywordAttr<IREE::CodeGen::PolicyMatchCriterion>(
          parser, "PolicyMatchCriterion", matchCriterion)) ||
      failed(parser.parseComma()) || failed(parser.parseAttribute(choices)) ||
      failed(parser.parseGreater())) {
    return {};
  }

  return OpPolicyAttr::get(op, builder.getStringAttr(matchCriterion), choices);
}

void OpPolicyAttr::print(DialectAsmPrinter &printer) const {
  auto &os = printer.getStream();
  os << getKindName() << "<";
  os << "\"" << op() << "\", " << match_criterion() << ", " << choices();
  os << ">";
}

Attribute TargetChoiceAttr::parse(DialectAsmParser &parser) {
  DictionaryAttr filter;
  ArrayAttr ops;

  if (failed(parser.parseLess()) || failed(parser.parseAttribute(filter)) ||
      failed(parser.parseArrow()) || failed(parser.parseAttribute(ops)) ||
      failed(parser.parseGreater())) {
    return {};
  }

  return TargetChoiceAttr::get(filter, ops);
}

void TargetChoiceAttr::print(DialectAsmPrinter &printer) const {
  auto &os = printer.getStream();
  os << getKindName() << "<" << filter() << " -> " << ops() << ">";
}

Attribute TargetPolicyAttr::parse(DialectAsmParser &parser) {
  StringAttr target;
  StringRef matchCriterion;
  ArrayAttr choices;

  if (failed(parser.parseLess()) || failed(parser.parseAttribute(target)) ||
      failed(parser.parseComma()) ||
      failed(parseEnumKeywordAttr<IREE::CodeGen::PolicyMatchCriterion>(
          parser, "PolicyMatchCriterion", matchCriterion)) ||
      failed(parser.parseComma()) || failed(parser.parseAttribute(choices)) ||
      failed(parser.parseGreater())) {
    return {};
  }

  return TargetPolicyAttr::get(
      target, parser.getBuilder().getStringAttr(matchCriterion), choices);
}

void TargetPolicyAttr::print(DialectAsmPrinter &printer) const {
  auto &os = printer.getStream();
  os << getKindName() << "<\"" << target() << "\", " << match_criterion()
     << ", " << choices() << ">";
}

}  // namespace CodeGen
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
