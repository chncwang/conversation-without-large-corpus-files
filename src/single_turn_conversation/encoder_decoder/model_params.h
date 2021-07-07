#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H

#include <fstream>
#include <iostream>

#include "insnet/insnet.h"

using namespace insnet;

struct ModelParams : public TunableParamCollection
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
    Embedding<Param> lookup_table;
    LinearParams hidden_to_wordvector_params;
    LinearParams hidden_to_keyword_params;
    LSTMParams l2r_encoder_params;
    LSTMParams r2l_encoder_params;
    LSTMParams decoder_params;
    AdditiveAttentionParams attention_params;

    ModelParams() : hidden_to_wordvector_params("hidden_to_wordvector_params"),
    hidden_to_keyword_params("hidden_to_keyword_params"), l2r_encoder_params("l2r_encoder"),
    r2l_encoder_params("r2l_encoder"), decoder_params("decoder"), attention_params("attention") {}

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(lookup_table, hidden_to_wordvector_params, hidden_to_keyword_params, l2r_encoder_params,
                r2l_encoder_params, decoder_params, attention_params);
    }

#if USE_GPU
    std::vector<cuda::Transferable *> transferablePtrs() override {
        return {&lookup_table, &hidden_to_wordvector_params, &hidden_to_keyword_params,
            &l2r_encoder_params, &r2l_encoder_params, &decoder_params, &attention_params};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam> *> tunableComponents() override {
        return {&lookup_table, &hidden_to_wordvector_params, &hidden_to_keyword_params,
            &l2r_encoder_params, &r2l_encoder_params, &decoder_params, &attention_params};
    }
};

#endif
