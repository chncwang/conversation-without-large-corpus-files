#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H

#include <fstream>
#include <iostream>

#include "N3LDG.h"

struct ModelParams : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
    LookupTable<Param> lookup_table;
    LookupTable<Param> idf_table;
    UniParams hidden_to_wordvector_params;
    ParamArray<UniParams> hidden_to_keyword_params;
    LSTM1Params left_to_right_encoder_params;
    LSTM1Params left_to_right_decoder_params;
    AdditiveAttentionParams attention_params;

    ModelParams() : hidden_to_wordvector_params("hidden_to_wordvector_params"),
    hidden_to_keyword_params("hidden_to_keyword_params"), left_to_right_encoder_params("encoder"),
    left_to_right_decoder_params("decoder"), attention_params("attention"){}

    Json::Value toJson() const override {
        Json::Value json;
        json["lookup_table"] = lookup_table.toJson();
        json["idf_table"] = idf_table.toJson();
        json["hidden_to_wordvector_params"] = hidden_to_wordvector_params.toJson();
        json["hidden_to_keyword_params"] = hidden_to_keyword_params.toJson();
        json["left_to_right_encoder_params"] = left_to_right_encoder_params.toJson();
        json["left_to_right_decoder_params"] = left_to_right_decoder_params.toJson();
        json["attention_params"] = attention_params.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        lookup_table.fromJson(json["lookup_table"]);
        idf_table.fromJson(json["idf_table"]);
        hidden_to_wordvector_params.fromJson(json["hidden_to_wordvector_params"]);
        hidden_to_keyword_params.fromJson(json["hidden_to_keyword_params"]);
        left_to_right_encoder_params.fromJson(json["left_to_right_encoder_params"]);
        left_to_right_decoder_params.fromJson(json["left_to_right_decoder_params"]);
        attention_params.fromJson(json["attention_params"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&lookup_table, &idf_table, &hidden_to_wordvector_params, &hidden_to_keyword_params,
            &left_to_right_encoder_params, &left_to_right_decoder_params, &attention_params};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam> *> tunableComponents() override {
        return {&lookup_table, &idf_table, &hidden_to_wordvector_params, &hidden_to_keyword_params,
            &left_to_right_encoder_params, &left_to_right_decoder_params, &attention_params};
    }
};

void initIdfTable(LookupTable<Param> &idf_embedding_table,
        const unordered_map<string, float> &idf_table) {
    for (const auto &it : idf_table) {
        int id = idf_embedding_table.elems.from_string(it.first);
        int dim = idf_embedding_table.E.outDim();
        for (int dim_i = 0; dim_i < dim; ++dim_i) {
            float idf = it.second;
            float x = idf / pow(1e2, static_cast<float>(dim_i) / dim);
            float y = 4 / (1 + exp(-x)) - 3;
            y = dim_i % 2 == 0 ? y : -y;
            idf_embedding_table.E.val[id][dim_i] = y;
        }
    }
#if USE_GPU
    idf_embedding_table.copyFromHostToDevice();
#endif
}

#endif
