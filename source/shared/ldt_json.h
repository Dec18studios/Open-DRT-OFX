#pragma once
//
// ldt_json.h — minimal dependency-free JSON reader for the Look Dev Tools suite.
//
// Vendored (hand-rolled) rather than pulled in as a build dependency: the
// control-channel schema is tiny and fixed, and the house rule for this repo is
// "no new external build deps".  Only the subset of JSON that the grade.json
// control file can contain is supported:
//
//   object, array, string, number, true/false/null
//
// Escapes handled in strings: \" \\ \/ \b \f \n \r \t and \uXXXX (encoded to
// UTF-8; surrogate pairs are combined).  Parsing is strict enough that a
// malformed file is rejected with a message rather than silently half-applied.
//
// Header-only, all functions inline.  No exceptions are thrown.
//
#include <string>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstdio>
#include <cmath>

namespace ldtjson {

enum class Type { Null, Bool, Number, String, Array, Object };

struct Value;
typedef std::vector<Value>            Array;
typedef std::map<std::string, Value>  Object;

struct Value
{
    Type        type = Type::Null;
    bool        boolean = false;
    double      number = 0.0;
    std::string str;
    Array       arr;
    Object      obj;

    bool isNull()   const { return type == Type::Null;   }
    bool isBool()   const { return type == Type::Bool;   }
    bool isNumber() const { return type == Type::Number; }
    bool isString() const { return type == Type::String; }
    bool isArray()  const { return type == Type::Array;  }
    bool isObject() const { return type == Type::Object; }

    // Object member lookup; returns nullptr when absent or when this is not an object.
    const Value* find(const std::string& key) const
    {
        if (type != Type::Object) return nullptr;
        Object::const_iterator it = obj.find(key);
        return (it == obj.end()) ? nullptr : &it->second;
    }

    // Convenience accessors used by the control-channel grade reader.
    bool asDouble(double& out) const
    {
        if (type == Type::Number) { out = number; return true; }
        return false;
    }
    bool asInt(long long& out) const
    {
        if (type == Type::Number) { out = (long long)llround(number); return true; }
        return false;
    }
};

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------
namespace detail {

struct Reader
{
    const char* p = nullptr;
    const char* end = nullptr;
    std::string error;

    bool fail(const char* msg)
    {
        if (error.empty()) {
            char buf[160];
            snprintf(buf, sizeof(buf), "%s at offset %ld", msg, (long)(p - beginPtr));
            error = buf;
        }
        return false;
    }

    const char* beginPtr = nullptr;

    void skipWs()
    {
        while (p < end) {
            const char c = *p;
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++p;
            else break;
        }
    }

    bool parseValue(Value& out, int depth);

    bool parseString(std::string& out)
    {
        if (p >= end || *p != '"') return fail("expected string");
        ++p;
        out.clear();
        while (true) {
            if (p >= end) return fail("unterminated string");
            unsigned char c = (unsigned char)*p;
            if (c == '"') { ++p; return true; }
            if (c == '\\') {
                ++p;
                if (p >= end) return fail("unterminated escape");
                char e = *p++;
                switch (e) {
                    case '"':  out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/');  break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        unsigned int cp = 0;
                        if (!readHex4(cp)) return false;
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            // High surrogate — expect a following \uXXXX low surrogate.
                            if (p + 1 < end && p[0] == '\\' && p[1] == 'u') {
                                p += 2;
                                unsigned int lo = 0;
                                if (!readHex4(lo)) return false;
                                if (lo >= 0xDC00 && lo <= 0xDFFF)
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                else
                                    cp = 0xFFFD;
                            } else {
                                cp = 0xFFFD;
                            }
                        }
                        appendUtf8(out, cp);
                        break;
                    }
                    default: return fail("bad escape");
                }
                continue;
            }
            if (c < 0x20) return fail("raw control character in string");
            out.push_back((char)c);
            ++p;
        }
    }

    bool readHex4(unsigned int& cp)
    {
        if (p + 4 > end) return fail("short \\u escape");
        cp = 0;
        for (int i = 0; i < 4; ++i) {
            char h = *p++;
            cp <<= 4;
            if (h >= '0' && h <= '9')      cp |= (unsigned)(h - '0');
            else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
            else return fail("bad hex digit in \\u escape");
        }
        return true;
    }

    static void appendUtf8(std::string& out, unsigned int cp)
    {
        if (cp < 0x80) {
            out.push_back((char)cp);
        } else if (cp < 0x800) {
            out.push_back((char)(0xC0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char)(0xE0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char)(0xF0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }

    bool parseNumber(Value& out)
    {
        const char* start = p;
        if (p < end && (*p == '-' || *p == '+')) ++p;
        bool any = false;
        while (p < end && *p >= '0' && *p <= '9') { ++p; any = true; }
        if (p < end && *p == '.') {
            ++p;
            while (p < end && *p >= '0' && *p <= '9') { ++p; any = true; }
        }
        if (!any) return fail("malformed number");
        if (p < end && (*p == 'e' || *p == 'E')) {
            ++p;
            if (p < end && (*p == '-' || *p == '+')) ++p;
            bool expAny = false;
            while (p < end && *p >= '0' && *p <= '9') { ++p; expAny = true; }
            if (!expAny) return fail("malformed exponent");
        }
        std::string tok(start, (size_t)(p - start));
        out.type   = Type::Number;
        out.number = strtod(tok.c_str(), nullptr);
        return true;
    }

    bool literal(const char* lit, size_t n)
    {
        if ((size_t)(end - p) < n) return false;
        for (size_t i = 0; i < n; ++i) if (p[i] != lit[i]) return false;
        p += n;
        return true;
    }
};

inline bool Reader::parseValue(Value& out, int depth)
{
    if (depth > 32) return fail("nesting too deep");
    skipWs();
    if (p >= end) return fail("unexpected end of input");

    const char c = *p;
    if (c == '{') {
        ++p;
        out = Value();
        out.type = Type::Object;
        skipWs();
        if (p < end && *p == '}') { ++p; return true; }
        while (true) {
            skipWs();
            std::string key;
            if (!parseString(key)) return false;
            skipWs();
            if (p >= end || *p != ':') return fail("expected ':'");
            ++p;
            Value v;
            if (!parseValue(v, depth + 1)) return false;
            out.obj[key] = v;
            skipWs();
            if (p < end && *p == ',') { ++p; continue; }
            if (p < end && *p == '}') { ++p; return true; }
            return fail("expected ',' or '}'");
        }
    }
    if (c == '[') {
        ++p;
        out = Value();
        out.type = Type::Array;
        skipWs();
        if (p < end && *p == ']') { ++p; return true; }
        while (true) {
            Value v;
            if (!parseValue(v, depth + 1)) return false;
            out.arr.push_back(v);
            skipWs();
            if (p < end && *p == ',') { ++p; continue; }
            if (p < end && *p == ']') { ++p; return true; }
            return fail("expected ',' or ']'");
        }
    }
    if (c == '"') {
        out = Value();
        out.type = Type::String;
        return parseString(out.str);
    }
    if (literal("true", 4))  { out = Value(); out.type = Type::Bool; out.boolean = true;  return true; }
    if (literal("false", 5)) { out = Value(); out.type = Type::Bool; out.boolean = false; return true; }
    if (literal("null", 4))  { out = Value(); out.type = Type::Null; return true; }

    out = Value();
    return parseNumber(out);
}

} // namespace detail

// Parse a complete JSON document.  Returns false and fills errorOut on failure.
inline bool parse(const std::string& text, Value& out, std::string& errorOut)
{
    detail::Reader r;
    r.p        = text.c_str();
    r.beginPtr = text.c_str();
    r.end      = text.c_str() + text.size();

    if (!r.parseValue(out, 0)) { errorOut = r.error; return false; }
    r.skipWs();
    if (r.p != r.end) { errorOut = "trailing characters after JSON document"; return false; }
    return true;
}

// ---------------------------------------------------------------------------
// Writing helpers (escaping only — LdtControl composes its output by hand)
// ---------------------------------------------------------------------------
inline std::string escape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                    out += buf;
                } else {
                    out.push_back((char)c);
                }
        }
    }
    return out;
}

} // namespace ldtjson
