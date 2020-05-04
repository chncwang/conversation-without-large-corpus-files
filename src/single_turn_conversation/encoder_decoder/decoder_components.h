#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "N3LDG.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

struct DecoderComponents {
    std::vector<Node *> decoder_lookups_before_dropout;
    std::vector<Node *> decoder_lookups;
    std::vector<Node *> decoder_to_wordvectors;
    std::vector<Node *> wordvector_to_onehots;
    DynamicLSTMBuilder decoder;
    vector<Node*> contexts;

    void forward(Graph &graph, const HyperParams &hyper_params, ModelParams &model_params,
            Node &input,
            vector<Node *> &encoder_hiddens,
            bool is_training) {
        using namespace n3ldg_plus;
        shared_ptr<AdditiveAttentionBuilder> attention_builder(new AdditiveAttentionBuilder);
        Node *hidden_bucket = bucket(graph, hyper_params.hidden_dim, 0.0f);
        Node *guide = decoder.size() == 0 ?  hidden_bucket :
            decoder._hiddens.at(decoder.size() - 1);

        attention_builder->forward(graph, model_params.attention_params, encoder_hiddens, *guide);
        contexts.push_back(attention_builder->_hidden);

        vector<Node *> ins = {&input, attention_builder->_hidden};
        Node *concat = n3ldg_plus::concat(graph, ins);

        decoder.forward(graph, model_params.left_to_right_decoder_params, *concat, *hidden_bucket,
                *hidden_bucket, hyper_params.dropout, is_training);
    }

    Node* decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            int i) {
        using namespace n3ldg_plus;
        vector<Node *> concat_inputs = {contexts.at(i), decoder._hiddens.at(i),
            i == 0 ? bucket(graph, hyper_params.word_dim, 0) :
                static_cast<Node*>(decoder_lookups.at(i - 1))};
        Node *concat_node = concat(graph, concat_inputs);

        Node *decoder_to_wordvector = n3ldg_plus::linear(graph,
                model_params.hidden_to_wordvector_params, *concat_node);
        return decoder_to_wordvector;
    }
};

#endif
