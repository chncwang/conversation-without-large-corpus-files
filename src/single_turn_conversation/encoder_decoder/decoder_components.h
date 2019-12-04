#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "N3LDG.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

struct ResultAndKeywordVectors {
    Node *result;
    Node *keyword;
    Node *keyword_extra;
};

struct DecoderComponents {
    std::vector<LookupNode<Param> *> decoder_lookups_before_dropout;
    std::vector<DropoutNode *> decoder_lookups;
    std::vector<LookupNode<Param> *> decoder_keyword_lookups;
    std::vector<Node *> decoder_to_wordvectors;
    std::vector<Node *> decoder_to_keyword_vectors;
    std::vector<Node *> wordvector_to_onehots;
    std::vector<Node *> keyword_vector_to_onehots;
    DynamicLSTMBuilder decoder;
    vector<Node*> contexts;

    BucketNode *bucket(int dim, Graph &graph) {
        BucketNode *node(new BucketNode);
        node->init(dim);
        node->forward(graph, 0);
        return node;
    }

    void forward(Graph &graph, const HyperParams &hyper_params, ModelParams &model_params,
            Node &input,
            Node &keyword_input,
            vector<Node *> &encoder_hiddens,
            bool is_training) {
        shared_ptr<AdditiveAttentionBuilder> attention_builder(new AdditiveAttentionBuilder);
        Node *guide = decoder.size() == 0 ?
            static_cast<Node*>(bucket(hyper_params.hidden_dim,
                        graph)) : static_cast<Node*>(decoder._hiddens.at(decoder.size() - 1));
        attention_builder->forward(graph, model_params.attention_params, encoder_hiddens, *guide);
        contexts.push_back(attention_builder->_hidden);

        ConcatNode* concat = new ConcatNode;
        concat->init(2 * hyper_params.word_dim + hyper_params.hidden_dim);
        vector<Node *> ins = {&input, &keyword_input, attention_builder->_hidden};
        concat->forward(graph, ins);

        decoder.forward(graph, model_params.left_to_right_encoder_params, *concat,
                *bucket(hyper_params.hidden_dim, graph),
                *bucket(hyper_params.hidden_dim, graph),
                hyper_params.dropout, is_training);
    }

    ResultAndKeywordVectors decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            vector<Node *> &encoder_hiddens,
            int i,
            bool return_keyword) {
        vector<Node *> concat_inputs = {
            contexts.at(i), decoder._hiddens.at(i),
            i == 0 ? bucket(hyper_params.word_dim, graph) :
                static_cast<Node*>(decoder_lookups.at(i - 1))
        };
        if (decoder_lookups.size() != i) {
            cerr << boost::format("decoder_lookups size:%1% i:%2%") % decoder_lookups.size() %
                i << endl;
            abort();
        }
        Node *concat_node = n3ldg_plus::concat(graph, concat_inputs);

        Node *keyword, *keyword_extra;
        if (return_keyword) {
            shared_ptr<AdditiveAttentionBuilder> keyword_att_builder(new AdditiveAttentionBuilder);
            Node *guide = decoder.size() == 0 ? static_cast<Node*>(bucket(hyper_params.hidden_dim,
                            graph)) : static_cast<Node*>(decoder._hiddens.at(decoder.size() - 1));
            keyword_att_builder->forward(graph, model_params.keyword_attention_params,
                    encoder_hiddens, *guide);
            Node *context_concated = n3ldg_plus::concat(graph,
                    {decoder._hiddens.at(i), contexts.at(i), keyword_att_builder->_hidden});
            keyword = n3ldg_plus::linear(graph, model_params.hidden_to_keyword_params,
                    *context_concated);
            keyword_extra = n3ldg_plus::linear(graph, model_params.hidden_to_keyword_extra_params,
                    *context_concated);
        } else {
            keyword = nullptr;
            keyword_extra = nullptr;
        }

        ConcatNode *keyword_concated = new ConcatNode();
        keyword_concated->init(concat_node->getDim() + hyper_params.word_dim);
        keyword_concated->forward(graph, {concat_node, decoder_keyword_lookups.at(i)});

        Node *decoder_to_wordvector = n3ldg_plus::linear(graph,
                model_params.hidden_to_wordvector_params, *keyword_concated);

        return {decoder_to_wordvector, keyword, keyword_extra};
    }
};

#endif
