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
    LSTMParams l2r_encoder_params;
    LSTMParams r2l_encoder_params;
    LSTMParams decoder_params;
    LinearParams output_params;
    AdditiveAttentionParams attention_params;

    ModelParams() : l2r_encoder_params("l2r_encoder_params"),
    r2l_encoder_params("r2l_encoder_params"), decoder_params("decoder_params"),
    output_params("output_params"), attention_params("attention_params") {}


    template<typename Archive>
    void serialize(Archive &ar) {
        ar(lookup_table, l2r_encoder_params, r2l_encoder_params, decoder_params, attention_params,
                output_params);
    }

#if USE_GPU
    std::vector<cuda::Transferable *> transferablePtrs() override {
        return {&lookup_table, &l2r_encoder_params, &r2l_encoder_params, &decoder_params,
            &attention_params, &output_params};
    }
#endif

protected:
    virtual std::vector<TunableParam *> tunableComponents() override {
        return {&lookup_table, &l2r_encoder_params, &r2l_encoder_params, &decoder_params,
            &attention_params, &output_params};
    }
};

#endif
