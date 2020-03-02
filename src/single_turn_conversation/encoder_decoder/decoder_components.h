#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "N3LDG.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

struct ResultAndKeywordVectors {
    Node *result;
    Node *keyword;
};

struct DecoderComponents {
    std::vector<Node *> decoder_lookups;
    std::vector<Node *> decoder_keyword_lookups;
    std::vector<Node *> decoder_to_wordvectors;
    std::vector<Node *> decoder_to_keyword_vectors;
    std::vector<Node *> wordvector_to_onehots;
    std::vector<Node *> keyword_vector_to_onehots;
    DynamicLSTMBuilder decoder;
    vector<Node*> contexts;

    void forward(Graph &graph, const HyperParams &hyper_params, ModelParams &model_params,
            Node &input,
            Node &keyword_input,
            vector<Node *> &encoder_hiddens,
            bool is_training) {
        shared_ptr<AdditiveAttentionBuilder> attention_builder(new AdditiveAttentionBuilder);
        Node *guide = decoder.size() == 0 ?
            static_cast<Node*>(n3ldg_plus::bucket(graph, hyper_params.hidden_dim, 0)) :
            static_cast<Node*>(decoder._hiddens.at(decoder.size() - 1));
        attention_builder->forward(graph, model_params.attention_params, encoder_hiddens, *guide);
        contexts.push_back(attention_builder->_hidden);

        vector<Node *> ins = {&input, &keyword_input, attention_builder->_hidden};
        Node *concat = n3ldg_plus::concat(graph, ins);

        decoder.forward(graph, model_params.left_to_right_decoder_params, *concat,
                *n3ldg_plus::bucket(graph, hyper_params.hidden_dim, 0),
                *n3ldg_plus::bucket(graph, hyper_params.hidden_dim, 0),
                hyper_params.dropout, is_training);
    }

    ResultAndKeywordVectors decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            vector<Node *> &encoder_hiddens,
            int i,
            bool return_keyword) {
        vector<Node *> concat_inputs = {
            contexts.at(i), decoder._hiddens.at(i),
            i == 0 ? n3ldg_plus::bucket(graph, hyper_params.word_dim, 0) :
                static_cast<Node*>(decoder_lookups.at(i - 1))
        };
        if (decoder_lookups.size() != i) {
            cerr << boost::format("decoder_lookups size:%1% i:%2%") % decoder_lookups.size() %
                i << endl;
            abort();
        }
        Node *concat_node = n3ldg_plus::concat(graph, concat_inputs);

        Node *keyword;
        if (return_keyword) {
            Node *context_concated = n3ldg_plus::concat(graph, {decoder._hiddens.at(i),
                    contexts.at(i)});
            keyword = n3ldg_plus::linear(graph, model_params.hidden_to_keyword_params,
                    *context_concated);
        } else {
            keyword = nullptr;
        }

        Node *keyword_concated = n3ldg_plus::concat(graph,
                {concat_node, decoder_keyword_lookups.at(i)});
        Node *decoder_to_wordvector = n3ldg_plus::linear(graph,
                model_params.hidden_to_wordvector_params, *keyword_concated);

        return {decoder_to_wordvector, keyword};
    }
};

#endif
