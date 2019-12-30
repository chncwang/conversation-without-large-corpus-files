#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H

#include <fstream>
#include <iostream>

#include "N3LDG.h"

class MLPParams : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents 
#endif
{
public:
    MLPParams(const string &name) : name_(name) {}

    void init(int layer, int row, int col, const std::function<float(int, int)> *bound = nullptr) {
        for (int i = 0; i < layer; ++i) {
            shared_ptr<UniParams> uni_params(new UniParams(name_ + std::to_string(i)));
            uni_params->init(row, i == 0 ? col : row, true, bound);
            keyword_mlp_params.push_back(uni_params);
        }
    }

    Json::Value toJson() const override {
        Json::Value json;
        for (const auto &p : keyword_mlp_params) {
            json.append(p->toJson());
        }
        return json;
    }

    void fromJson(const Json::Value &json) override {
        int i = 0;
        for (const auto &p : keyword_mlp_params) {
            p->fromJson(json[i++]);
        }
    }

    UniParams *get(int i) const {
        return keyword_mlp_params.at(i).get();
    }

    int layerSize() const {
        return keyword_mlp_params.size();
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        std::vector<Transferable *> ptrs;
        transform(keyword_mlp_params.begin(), keyword_mlp_params.end(), back_inserter(ptrs),
                [](const shared_ptr<UniParams> &in) { return in.get(); });
        return ptrs;
    }

    virtual std::string name() const {
        return "MLPParams";
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        vector<Tunable<BaseParam> *> results;
        transform(keyword_mlp_params.begin(), keyword_mlp_params.end(), back_inserter(results),
                [](const shared_ptr<UniParams> &in) { return in.get();});
        return results;
    }

private:
    vector<shared_ptr<UniParams>> keyword_mlp_params;
    string name_;
};

Node *mlp(Graph &graph, MLPParams &params, Node &input) {
    Node *node = n3ldg_plus::linear(graph, *params.get(0), input);
    for (int i = 1; i < params.layerSize(); ++i) {
        Node *next = n3ldg_plus::uni(graph, *params.get(i), *node, ActivatedEnum::RELU);
        node = n3ldg_plus::add(graph, {node, next});
    }

    return node;
}

struct ModelParams : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
    LookupTable<Param> lookup_table;
    UniParams hidden_to_wordvector_params;
    UniParams hidden_to_keyword_params;
    LSTM1Params left_to_right_encoder_params;
    LSTM1Params left_to_right_decoder_params;
    AdditiveAttentionParams attention_params;
    MLPParams keyword_mlp_params;

    ModelParams() : hidden_to_wordvector_params("hidden_to_wordvector_params"),
    hidden_to_keyword_params("hidden_to_keyword_params"), left_to_right_encoder_params("encoder"),
    left_to_right_decoder_params("decoder"), attention_params("attention"),
    keyword_mlp_params("keyword_mlp_params"){}

    Json::Value toJson() const override {
        Json::Value json;
        json["lookup_table"] = lookup_table.toJson();
        json["hidden_to_wordvector_params"] = hidden_to_wordvector_params.toJson();
        json["hidden_to_keyword_params"] = hidden_to_keyword_params.toJson();
        json["left_to_right_encoder_params"] = left_to_right_encoder_params.toJson();
        json["left_to_right_decoder_params"] = left_to_right_decoder_params.toJson();
        json["attention_params"] = attention_params.toJson();
        json["keyword_mlp_params"] = keyword_mlp_params.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        lookup_table.fromJson(json["lookup_table"]);
        hidden_to_wordvector_params.fromJson(json["hidden_to_wordvector_params"]);
        hidden_to_keyword_params.fromJson(json["hidden_to_keyword_params"]);
        left_to_right_encoder_params.fromJson(json["left_to_right_encoder_params"]);
        left_to_right_decoder_params.fromJson(json["left_to_right_decoder_params"]);
        attention_params.fromJson(json["attention_params"]);
        keyword_mlp_params.fromJson(json["keyword_mlp_params"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&lookup_table, &hidden_to_wordvector_params, &hidden_to_keyword_params,
            &left_to_right_encoder_params, &left_to_right_decoder_params, &attention_params,
            &keyword_mlp_params};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam> *> tunableComponents() override {
        return {&lookup_table, &hidden_to_wordvector_params, &hidden_to_keyword_params,
            &left_to_right_encoder_params, &left_to_right_decoder_params, &attention_params,
            &keyword_mlp_params};
    }
};

#endif
