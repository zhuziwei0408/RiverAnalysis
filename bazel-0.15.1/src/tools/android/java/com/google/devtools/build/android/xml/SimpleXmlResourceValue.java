// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.xml;

import com.android.aapt.Resources.Item;
import com.android.aapt.Resources.StyledString;
import com.android.aapt.Resources.StyledString.Span;
import com.android.aapt.Resources.Value;
import com.android.resources.ResourceType;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableMap;
import com.google.common.xml.XmlEscapers;
import com.google.devtools.build.android.AndroidDataWritingVisitor;
import com.google.devtools.build.android.AndroidDataWritingVisitor.StartTag;
import com.google.devtools.build.android.AndroidResourceSymbolSink;
import com.google.devtools.build.android.DataSource;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;
import com.google.devtools.build.android.XmlResourceValues;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.DataValueXml;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;
import javax.xml.namespace.QName;

/**
 * Represents a simple Android resource xml value.
 *
 * <p>
 * There is a class of resources that are simple name/value pairs: string
 * (http://developer.android.com/guide/topics/resources/string-resource.html), bool
 * (http://developer.android.com/guide/topics/resources/more-resources.html#Bool), color
 * (http://developer.android.com/guide/topics/resources/more-resources.html#Color), and dimen
 * (http://developer.android.com/guide/topics/resources/more-resources.html#Dimension). These are
 * defined in xml as &lt;<em>resource type</em> name="<em>name</em>" value="<em>value</em>"&gt;. In
 * the interest of keeping the parsing svelte, these are represented by a single class.
 */
@Immutable
public class SimpleXmlResourceValue implements XmlResourceValue {
  static final QName TAG_BOOL = QName.valueOf("bool");
  static final QName TAG_COLOR = QName.valueOf("color");
  static final QName TAG_DIMEN = QName.valueOf("dimen");
  static final QName TAG_DRAWABLE = QName.valueOf("drawable");
  static final QName TAG_FRACTION = QName.valueOf("fraction");
  static final QName TAG_INTEGER = QName.valueOf("integer");
  static final QName TAG_ITEM = QName.valueOf("item");
  static final QName TAG_LAYOUT = QName.valueOf("layout");
  static final QName TAG_MENU = QName.valueOf("menu");
  static final QName TAG_MIPMAP = QName.valueOf("mipmap");
  static final QName TAG_PUBLIC = QName.valueOf("public");
  static final QName TAG_RAW = QName.valueOf("raw");
  static final QName TAG_STRING = QName.valueOf("string");

  /** Provides an enumeration resource type and simple value validation. */
  public enum Type {
    BOOL(TAG_BOOL) {
      @Override
      public boolean validate(String value) {
        final String cleanValue = value.toLowerCase().trim();
        return "true".equals(cleanValue) || "false".equals(cleanValue);
      }
    },
    COLOR(TAG_COLOR) {
      @Override
      public boolean validate(String value) {
        // TODO(corysmith): Validate the hex color.
        return true;
      }
    },
    DIMEN(TAG_DIMEN) {
      @Override
      public boolean validate(String value) {
        // TODO(corysmith): Validate the dimension type.
        return true;
      }
    },
    DRAWABLE(TAG_DRAWABLE) {
      @Override
      public boolean validate(String value) {
        // TODO(corysmith): Validate the drawable type.
        return true;
      }
    },
    FRACTION(TAG_FRACTION) {
      @Override
      public boolean validate(String value) {
        // TODO(corysmith): Validate the fraction type.
        return true;
      }
    },
    INTEGER(TAG_INTEGER) {
      @Override
      public boolean validate(String value) {
        // TODO(corysmith): Validate the integer type.
        return true;
      }
    },
    ITEM(TAG_ITEM) {
      @Override
      public boolean validate(String value) {
        // TODO(corysmith): Validate the item type.
        return true;
      }
    },
    LAYOUT(TAG_LAYOUT) {
      @Override
      public boolean validate(String value) {
        // TODO(corysmith): Validate the layout type.
        return true;
      }
    },
    MENU(TAG_MENU) {
      @Override
      public boolean validate(String value) {
        // TODO(corysmith): Validate the menu type.
        return true;
      }
    },
    MIPMAP(TAG_MIPMAP) {
      @Override
      public boolean validate(String value) {
        // TODO(corysmith): Validate the mipmap type.
        return true;
      }
    },
    PUBLIC(TAG_PUBLIC) {
      @Override
      public boolean validate(String value) {
        // TODO(corysmith): Validate the public type.
        return true;
      }
    },
    RAW(TAG_RAW) {
      @Override
      public boolean validate(String value) {
        // TODO(corysmith): Validate the raw type.
        return true;
      }
    },
    STRING(TAG_STRING) {
      @Override
      public boolean validate(String value) {
        return true;
      }
    };
    private final QName tagName;

    Type(QName tagName) {
      this.tagName = tagName;
    }

    abstract boolean validate(String value);

    public static Type from(ResourceType resourceType) {
      for (Type valueType : values()) {
        if (valueType.tagName.getLocalPart().equals(resourceType.getName())) {
          return valueType;
        } else if (resourceType.getName().equalsIgnoreCase(valueType.name())) {
          return valueType;
        }
      }
      throw new IllegalArgumentException(
          String.format(
              "%s resource type not found in available types: %s",
              resourceType,
              Arrays.toString(values())));
    }
  }

  private final ImmutableMap<String, String> attributes;
  @Nullable private final String value;
  private final Type valueType;

  public static XmlResourceValue createWithValue(Type valueType, String value) {
    return of(valueType, ImmutableMap.<String, String>of(), value);
  }

  public static XmlResourceValue withAttributes(
      Type valueType, ImmutableMap<String, String> attributes) {
    return of(valueType, attributes, null);
  }

  public static XmlResourceValue itemWithFormattedValue(
      ResourceType resourceType, String format, String value) {
    return of(Type.ITEM, ImmutableMap.of("type", resourceType.getName(), "format", format), value);
  }

  public static XmlResourceValue itemWithValue(
      ResourceType resourceType, String value) {
    return of(Type.ITEM, ImmutableMap.of("type", resourceType.getName()), value);
  }

  public static XmlResourceValue itemPlaceHolderFor(ResourceType resourceType) {
    return withAttributes(Type.ITEM, ImmutableMap.of("type", resourceType.getName()));
  }

  public static XmlResourceValue of(
      Type valueType, ImmutableMap<String, String> attributes, @Nullable String value) {
    return new SimpleXmlResourceValue(valueType, attributes, value);
  }

  private SimpleXmlResourceValue(
      Type valueType, ImmutableMap<String, String> attributes, String value) {
    this.valueType = valueType;
    this.value = value;
    this.attributes = attributes;
  }

  @Override
  public void write(
      FullyQualifiedName key, DataSource source, AndroidDataWritingVisitor mergedDataWriter) {

    StartTag startTag =
        mergedDataWriter
            .define(key)
            .derivedFrom(source)
            .startTag(valueType.tagName)
            .named(key)
            .addAttributesFrom(attributes.entrySet());

    if (value != null) {
      startTag.closeTag().addCharactersOf(value).endTag().save();
    } else {
      startTag.closeUnaryTag().save();
    }
  }

  @SuppressWarnings("deprecation")
  public static XmlResourceValue from(SerializeFormat.DataValueXml proto) {
    return of(
        Type.valueOf(proto.getValueType()),
        ImmutableMap.copyOf(proto.getAttribute()),
        proto.hasValue() ? proto.getValue() : null);
  }

  public static XmlResourceValue from(Value proto, ResourceType resourceType) {
    Item item = proto.getItem();
    String stringValue = null;

    if (item.hasStr()) {
      stringValue = XmlEscapers.xmlContentEscaper().escape(item.getStr().getValue());
    } else if (item.hasRef()) {
      stringValue = "@" + item.getRef().getName();
    } else if (item.hasStyledStr()) {
      StyledString styledString = item.getStyledStr();
      StringBuilder stringBuilder = new StringBuilder(styledString.getValue());

      for (Span span : styledString.getSpanList()) {
        stringBuilder.append(
            String.format(";%s,%d,%d", span.getTag(), span.getFirstChar(), span.getLastChar()));
      }
      stringValue = stringBuilder.toString();
    } else if ((resourceType == ResourceType.COLOR
        || resourceType == ResourceType.DRAWABLE) && item.hasPrim()) {
      stringValue =
          String.format("#%1$8s", Integer.toHexString(item.getPrim().getData())).replace(' ', '0');
    } else if (resourceType == ResourceType.INTEGER && item.hasPrim()){
      stringValue = Integer.toString(item.getPrim().getData());
    } else if (resourceType == ResourceType.BOOL && item.hasPrim()) {
      stringValue = item.getPrim().getData() == 0 ? "false" : "true";
    } else if (resourceType == ResourceType.FRACTION
        || resourceType == ResourceType.DIMEN
        || resourceType == ResourceType.STRING) {
      stringValue = Integer.toString(item.getPrim().getData());
    } else {
      throw new IllegalArgumentException(
          String.format("'%s' with value %s is not a simple resource type.", resourceType, proto));
    }

    return of(
        Type.valueOf(resourceType.toString().toUpperCase(Locale.ENGLISH)),
        ImmutableMap.of(),
        stringValue);
  }

  @Override
  public void writeResourceToClass(FullyQualifiedName key, AndroidResourceSymbolSink sink) {
    sink.acceptSimpleResource(key.type(), key.name());
  }

  @Override
  public int serializeTo(int sourceId, Namespaces namespaces, OutputStream output)
      throws IOException {
    SerializeFormat.DataValue.Builder builder =
        XmlResourceValues.newSerializableDataValueBuilder(sourceId);
    DataValueXml.Builder xmlValueBuilder =
        builder
            .getXmlValueBuilder()
            .putAllNamespace(namespaces.asMap())
            .setType(SerializeFormat.DataValueXml.XmlType.SIMPLE)
            // TODO(corysmith): Find a way to avoid writing strings to the serialized format
            // it's inefficient use of space and costs more when deserializing.
            .putAllAttribute(attributes);
    if (value != null) {
      xmlValueBuilder.setValue(value);
    }
    builder.setXmlValue(xmlValueBuilder.setValueType(valueType.name()));
    return XmlResourceValues.serializeProtoDataValue(output, builder);
  }

  @Override
  public int hashCode() {
    return Objects.hash(valueType, attributes, value);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof SimpleXmlResourceValue)) {
      return false;
    }
    SimpleXmlResourceValue other = (SimpleXmlResourceValue) obj;
    return Objects.equals(valueType, other.valueType)
        && Objects.equals(attributes, other.attributes)
        && Objects.equals(value, other.value);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("valueType", valueType)
        .add("attributes", attributes)
        .add("value", value)
        .toString();
  }

  @Override
  public XmlResourceValue combineWith(XmlResourceValue value) {
    throw new IllegalArgumentException(this + " is not a combinable resource.");
  }
  
  @Override
  public String asConflictStringWith(DataSource source) {
    if (value != null) {
      return String.format(" %s (with value %s)", source.asConflictString(), value);
    }
    return source.asConflictString();
  }
}
