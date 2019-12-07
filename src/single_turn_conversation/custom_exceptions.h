#ifndef SINGLE_TURN_CONVERSATION_SRC_CUSTOM_EXCEPTIONS_H
#define SINGLE_TURN_CONVERSATION_SRC_CUSTOM_EXCEPTIONS_H

#include <exception>
#include <json/json.h>
#include <string>

class InformedRuntimeError : public std::exception {
public:
    InformedRuntimeError(const Json::Value &info) : info_(info) {}

    const Json::Value &getInfo() const {
        return info_;
    }

private:
    Json::Value info_;
};

#endif
