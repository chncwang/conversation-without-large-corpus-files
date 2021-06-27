#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "insnet/insnet.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

using namespace insnet;

Node* decoderToWordVectors(const std::vector<Node *> &hiddens, int dec_sentence_len,
        const HyperParams &hyper_params,
        ModelParams &model_params) {
    using namespace insnet;
    Node *hidden = cat(hiddens);
    return hidden;
}

struct DecoderCellComponents {
    Node *wordvector_to_onehot;
    LSTMState state;

    DecoderCellComponents() = default;

    Node* decoderToWordVectors(const HyperParams &hyper_params, ModelParams &model_params) {
        using namespace insnet;
        return state.hidden;
    }
};
#endif
