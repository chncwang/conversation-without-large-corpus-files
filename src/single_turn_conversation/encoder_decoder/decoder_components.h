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
    vector<DynamicLSTMBuilder> decoders;
    vector<Node*> contexts;

    void init(int layer) {
        decoders.resize(layer);
    }

    void forward(Graph &graph, const HyperParams &hyper_params, ModelParams &model_params,
            Node &input,
            Node &keyword_input,
            vector<Node *> &encoder_hiddens,
            bool is_training) {
        shared_ptr<AdditiveAttentionBuilder> attention_builder(new AdditiveAttentionBuilder);
        Node *guide;
        if (decoders.at(0).size() == 0) {
            guide = n3ldg_plus::bucket(graph, hyper_params.hidden_dim, 0);
        } else {
            int prev_i = decoders.at(0).size() - 1;
            vector<Node *> ins;
            for (auto &decoder : decoders) {
                ins.push_back(decoder._hiddens.at(prev_i));
            }
            guide = n3ldg_plus::add(graph, ins);
            guide = n3ldg_plus::dropout(graph, *guide, 1 - 1.0 / ins.size(), false);
        }

        attention_builder->forward(graph, model_params.attention_params, encoder_hiddens, *guide);
        contexts.push_back(attention_builder->_hidden);
        vector<Node *> ins = {&input, &keyword_input, attention_builder->_hidden};
        Node *concat = n3ldg_plus::concat(graph, ins);
        Node *last_input = n3ldg_plus::linear(graph, model_params.decoder_input_linear_params,
                *concat);

        for (int i = 0; i < hyper_params.decoder_layer; ++i) {
            decoders.at(i).forward(graph, *model_params.left_to_right_decoder_params.ptrs().at(i),
                    *last_input, *n3ldg_plus::bucket(graph, hyper_params.hidden_dim, 0),
                    *n3ldg_plus::bucket(graph, hyper_params.hidden_dim, 0),
                    hyper_params.dropout, is_training);
            if (i < hyper_params.decoder_layer - 1) {
                last_input = n3ldg_plus::add(graph, {last_input, decoders.at(i)._hiddens.back()});
            }
        }
    }

    ResultAndKeywordVectors decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            vector<Node *> &encoder_hiddens,
            int i,
            bool return_keyword) {
        vector<Node *> concat_inputs = {
            contexts.at(i), decoders.back()._hiddens.at(i),
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
            Node *context_concated = n3ldg_plus::concat(graph, {decoders.back()._hiddens.at(i),
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
