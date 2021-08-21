#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "insnet/insnet.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

using namespace insnet;

struct DecoderCellComponents {
    Node *normal_prob = nullptr;
    Node *normal_emb = nullptr;
    Node *keyword_emb = nullptr;
    Node *context = nullptr;
    LSTMState state;
    int last_word_id = 0;
    int last_keyword_id = 0;

    DecoderCellComponents() = default;
};

#endif

