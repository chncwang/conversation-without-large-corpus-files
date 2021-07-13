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
    LinearParams hidden_to_wordvector_params_a;
    LinearParams hidden_to_wordvector_params_b;
    LinearParams hidden_to_keyword_params_a;
    LinearParams hidden_to_keyword_params_b;
    TransformerEncoderParams encoder_params;
    TransformerDecoderParams decoder_params;

    ModelParams() : hidden_to_wordvector_params_a("hidden_to_wordvector_params_a"),
    hidden_to_wordvector_params_b("hidden_to_wordvector_params_b"),
    hidden_to_keyword_params_a("hidden_to_keyword_params_a"),
    hidden_to_keyword_params_b("hidden_to_keyword_params_b"),
    encoder_params("encoder_params"), decoder_params("decoder") {}

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(lookup_table, hidden_to_wordvector_params_a, hidden_to_wordvector_params_b,
                hidden_to_keyword_params_a, hidden_to_keyword_params_b, encoder_params,
                decoder_params);
    }

#if USE_GPU
    std::vector<cuda::Transferable *> transferablePtrs() override {
        return {&lookup_table, &hidden_to_wordvector_params_a, &hidden_to_wordvector_params_b,
            &hidden_to_keyword_params_a, &hidden_to_keyword_params_b, &encoder_params,
            &decoder_params};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam> *> tunableComponents() override {
        return {&lookup_table, &hidden_to_wordvector_params_a, &hidden_to_wordvector_params_b,
            &hidden_to_keyword_params_a, &hidden_to_keyword_params_b, &encoder_params,
            &decoder_params};
    }
};

#endif
