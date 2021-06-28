#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H

#include <cmath>
#include <vector>
#include <array>
#include <set>
#include <string>
#include <memory>
#include <tuple>
#include <queue>
#include <algorithm>
#include <boost/format.hpp>
#include "insnet/insnet.h"
#include "tinyutf8.h"
#include "model_params.h"
#include "hyper_params.h"
#include "single_turn_conversation/default_config.h"
#include "single_turn_conversation/encoder_decoder/decoder_components.h"
#include "single_turn_conversation/def.h"

using namespace std;

set<std::array<int, 3>> triSet(const vector<int> &sentence) {
    if (sentence.size() < 3) {
        cerr << "triSet" << endl;
        abort();
    }
    using std::array;
    set<array<int, 3>> results;
    for (int i = 0; i < sentence.size() - 2; ++i) {
        array<int, 3> tri = {sentence.at(i + 0), sentence.at(i + 1), sentence.at(i + 2)};
        results.insert(tri);
    }

    if (results.size() != sentence.size() - 2) {
        cerr << boost::format("triSet - result size is %1%, but sentence len is %2%") %
            results.size() % sentence.size() << endl;
        abort();
    }

    return results;
}

set<int> repeatedIds(const vector<int> &sentence) {
    auto tri_set = triSet(sentence);
    set<int> results;
    for (const auto &tri : tri_set) {
        int sentence_len = sentence.size();
        if (tri.at(0) == sentence.at(sentence_len - 2) &&
                tri.at(1) == sentence.at(sentence_len - 1)) {
            results.insert(tri.at(2));
        }
    }
    return results;
}

string getSentence(const vector<int> &word_ids_vector, const ModelParams &model_params) {
    string words;
    for (const int &w : word_ids_vector) {
        string str = model_params.lookup_table.vocab.from_id(w);
        words += str;
    }
    return words;
}

#define BEAM_SEARCH_KEY "beam_search"

class BeamSearchResult {
public:
    BeamSearchResult() = default;

    BeamSearchResult(const BeamSearchResult &beam_search_result) = default;

    BeamSearchResult(const DecoderCellComponents &decoder_components, const vector<int> &pathh,
            dtype log_probability) : decoder_components_(decoder_components), path_(pathh),
            final_log_probability(log_probability) {}

    dtype finalScore() const {
        return (final_log_probability ) / path_.size();
    }

    dtype finalLogProbability() const {
        return final_log_probability;
    }

    const vector<int> &getPath() const {
        return path_;
    }

    const DecoderCellComponents &decoderComponents() const {
        return decoder_components_;
    }

private:
    DecoderCellComponents decoder_components_;
    vector<int> path_;
    dtype final_log_probability;
};

void printWordIds(const vector<int> &word_ids_with_probability_vector,
        const Embedding<Param> &lookup_table,
        bool print_space = true) {
    for (const int &id : word_ids_with_probability_vector) {
        cout << lookup_table.vocab.from_id(id);
        if (print_space && &id != &word_ids_with_probability_vector.back()) {
            cout << " ";
        }
    }
    cout << endl;
}

int countNgramDuplicate(const vector<int> &ids, int n) {
    if (n >= ids.size()) {
        return 0;
    }
    vector<int> target;
    for (int i = 0; i < n; ++i) {
        target.push_back(ids.at(ids.size() - n + i));
    }

    int duplicate_count = 0;

    for (int i = 0; i < ids.size() - n; ++i) {
        bool same = true;
        for (int j = 0; j < n; ++j) {
            if (target.at(j) != ids.at(i + j)) {
                same = false;
                break;
            }
        }
        if (same) {
            ++duplicate_count;
        }
    }

    return duplicate_count;
}

struct QueueEle {
    int word_id;
    int source;
    dtype log_prob;
};

vector<BeamSearchResult> mostProbableResults(
        const vector<shared_ptr<DecoderCellComponents>> &beam,
        const vector<BeamSearchResult> &last_results,
        int current_word,
        int k,
        const ModelParams &model_params,
        const DefaultConfig &default_config,
        bool is_first,
        bool check_tri_gram,
        const vector<string> &black_list,
        set<int> &searched_word_ids,
        Graph &graph) {
    vector<Node *> nodes;
    for (const auto &decoder_components : beam) {
        nodes.push_back(decoder_components->wordvector_to_onehot);
        if (is_first) {
            break;
        }
    }
    if (!is_first && nodes.size() != last_results.size() && !last_results.empty()) {
        cerr << boost::format(
                "nodes size is not equal to last_results size, nodes size is %1% but last_results size is %2%")
            % nodes.size() % last_results.size() << endl;
        abort();
    }

    auto cmp = [&](const QueueEle &a, const QueueEle &b) {
        return a.log_prob > b.log_prob;
    };
    priority_queue<QueueEle, vector<QueueEle>, decltype(cmp)> queue(cmp);
    for (int i = 0; i < (is_first ? 1 : nodes.size()); ++i) {
        Node &node = *nodes.at(i);
#if USE_GPU
        node.val().initOnMemory(node.size());
        node.val().copyFromDeviceToHost();
#endif
        set<int> repeated_ids;
        if (check_tri_gram) {
            vector<int> word_ids;
            for (int id : last_results.at(i).getPath()) {
                word_ids.push_back(id);
            }
            repeated_ids = repeatedIds(word_ids);
        }
        for (int j = 0; j < nodes.at(i)->size(); ++j) {
            if (repeated_ids.find(j) != repeated_ids.end()) {
                continue;
            }
            if (is_first) {
                if (searched_word_ids.find(j) != searched_word_ids.end()) {
                    cout << boost::format("word id searched:%1% word:%2%\n") % j %
                        model_params.lookup_table.vocab.from_id(j);
                    continue;
                }
            }
            if (j == model_params.lookup_table.getElemId(insnet::UNKNOWN_WORD)) {
                continue;
            }
            dtype word_probability = node.getVal().v[j];
            dtype log_probability = log(word_probability);
            if (!last_results.empty()) {
                log_probability += last_results.at(i).finalLogProbability();
            }

            if (queue.size() < k) {
                queue.push({j, i, log_probability});
            } else if (queue.top().log_prob < log_probability) {
                queue.pop();
                queue.push({j, i, log_probability});
            }
        }
    }

    vector<BeamSearchResult> results;
    cout << "queue size:" << queue.size() << endl;
    while (!queue.empty()) {
        auto &e = queue.top();
        if (is_first) {
            int source = e.source;
            int first_id = last_results.empty() ? e.word_id :
                last_results.at(source).getPath().front();
            searched_word_ids.insert(first_id);
        }
        vector<int> last_path;
        if (!last_results.empty()) {
            last_path = last_results.at(e.source).getPath();
        }
        last_path.push_back(e.word_id);

        BeamSearchResult result(*beam.at(e.source), last_path, e.log_prob);
        results.push_back(result);
        queue.pop();
    }

    vector<BeamSearchResult> final_results;
    int i = 0;
    for (const BeamSearchResult &result : results) {
        const auto &ids = result.getPath();
        string sentence = ::getSentence(ids, model_params);
        bool contain_black = false;
        for (const string &str : black_list) {
            utf8_string utf8_str(str), utf8_sentece(sentence);
            if (utf8_sentece.find(utf8_str) != string::npos) {
                contain_black = true;
                break;
            }
        }
        if (contain_black) {
            continue;
        }
        final_results.push_back(result);
        cout << boost::format("mostProbableResults - i:%1% prob:%2% score:%3%") % i %
            result.finalLogProbability() % result.finalScore() << endl;
        printWordIds(result.getPath(), model_params.lookup_table);
        ++i;
    }

    return final_results;
}

struct GraphBuilder {
    Node *encoder_hiddens;
    int enc_len;

    void forward(Graph &graph, const vector<string> &sentence,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        using namespace insnet;
        vector<Node *> embs;
        for (const string &w : sentence) {
            Node *emb = embedding(graph, w, model_params.lookup_table);
            emb = dropout(*emb, hyper_params.dropout);
            embs.push_back(emb);
        }
        Node *h0 = tensor(graph, hyper_params.hidden_dim, 0.0f);
        LSTMState initial_state = {h0, h0};
        std::vector<Node *> l2r = lstm(initial_state, embs, model_params.l2r_encoder_params,
                hyper_params.dropout);
        std::reverse(embs.begin(), embs.end());
        std::vector<Node *> r2l = lstm(initial_state, embs, model_params.r2l_encoder_params,
                hyper_params.dropout);
        std::reverse(r2l.begin(), r2l.end());

        Node *l2r_matrix = cat(l2r);
        Node *r2l_matrix = cat(r2l);
        encoder_hiddens = cat({l2r_matrix, r2l_matrix}, l2r.size());
        enc_len = l2r.size();
    }

    Node *forwardDecoder(const vector<string> &answer, const HyperParams &hyper_params,
            ModelParams &model_params) {
        using namespace insnet;

        vector<string> words;
        words.push_back(BEGIN_SYMBOL);
        for (int i = 1; i < answer.size(); ++i) {
            words.push_back(answer.at(i - 1));
        }
        Graph &graph = dynamic_cast<Graph &>(encoder_hiddens->getNodeContainer());
        Node *h0 = tensor(graph, hyper_params.hidden_dim, 0.0f);
        LSTMState last_state = {h0, h0};
        vector<Node *> decoder_hiddens;
        for (const string &w : words) {
            Node *emb = embedding(graph, w, model_params.lookup_table);
            emb = dropout(*emb, hyper_params.dropout);
            Node *context = additiveAttention(*last_state.hidden, *encoder_hiddens, enc_len,
                    model_params.attention_params).first;
            Node *in = cat({emb, context});
            last_state = lstm(last_state, *in, model_params.decoder_params, hyper_params.dropout);
            decoder_hiddens.push_back(last_state.hidden);
        }

        Node *hidden_matrix = cat(decoder_hiddens);
        Node *hidden_linear = linear(*hidden_matrix, model_params.output_params);
        Node *onehot = linear(*hidden_linear, model_params.lookup_table.E);
        Node *softmax = insnet::softmax(*onehot, model_params.lookup_table.size());
        return softmax;
    }

    void forwardDecoderByOneStep(Graph &graph, DecoderCellComponents &decoder_components,
            const string &answer,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        using namespace insnet;
        Node *emb = embedding(graph, answer, model_params.lookup_table);
        decoder_components.state = lstm(decoder_components.state, *emb,
                model_params.decoder_params, hyper_params.dropout);
        Node *decoder_to_wordvector = decoder_components.decoderToWordVectors(hyper_params,
                model_params);
        Node *onehot = linear(*decoder_to_wordvector, model_params.lookup_table.E);
        Node *softmax = insnet::softmax(*onehot, onehot->size());
        decoder_components.wordvector_to_onehot = softmax;
    }

    pair<vector<int>, dtype> forwardDecoderUsingBeamSearch(Graph &graph,
            const vector<shared_ptr<DecoderCellComponents>> &decoder_components_beam,
            int k,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            const DefaultConfig &default_config,
            const vector<string> &black_list) {
        vector<pair<vector<int>, dtype>> word_ids_result;
        vector<BeamSearchResult> most_probable_results;
        vector<string> last_answers;
        bool succeeded = false;
        set<int> searched_word_ids;

        for (int iter = 0; ; ++iter) {
            cout << boost::format("forwardDecoderUsingBeamSearch iter:%1%\n") % iter;
            most_probable_results.clear();
            auto beam = decoder_components_beam;
            cout << boost::format("beam size:%1%\n") % beam.size();

            int ended_count = word_ids_result.size();
            if (ended_count >= default_config.result_count_factor * k) {
                break;
            }

            for (int i = 0;; ++i) {
                cout << boost::format("forwardDecoderUsingBeamSearch i:%1%\n") % i;
                if (word_ids_result.size() >= default_config.result_count_factor * k ||
                        i > default_config.cut_length) {
                    break;
                }

                last_answers.clear();
                if (i > 0) {
                    most_probable_results = mostProbableResults(beam, most_probable_results, i,
                            k, model_params, default_config, i == 1, i >= 4, black_list,
                            searched_word_ids, graph);
                    cout << boost::format("most_probable_results size:%1%") %
                        most_probable_results.size() << endl;
                    auto last_beam = beam;
                    beam.clear();
                    vector<BeamSearchResult> stop_removed_results;
                    int j = 0;
                    for (BeamSearchResult &beam_search_result : most_probable_results) {
                        const vector<int> &word_ids = beam_search_result.getPath();

                        int last_word_id = word_ids.at(word_ids.size() - 1);
                        const string &word = model_params.lookup_table.vocab.from_id(last_word_id);
                        if (word == STOP_SYMBOL) {
                            word_ids_result.push_back(make_pair(word_ids,
                                        beam_search_result.finalScore()));
                            succeeded = word == STOP_SYMBOL;
                        } else {
                            stop_removed_results.push_back(beam_search_result);
                            last_answers.push_back(word);
                            shared_ptr<DecoderCellComponents> ptr(new DecoderCellComponents(
                                        beam_search_result.decoderComponents()));
                            beam.push_back(ptr);
                        }
                        ++j;
                    }
                    most_probable_results = stop_removed_results;
                }

                if (beam.empty()) {
                    cout << boost::format("break for beam empty\n");
                    break;
                }

                for (int beam_i = 0; beam_i < beam.size(); ++beam_i) {
                    DecoderCellComponents &decoder_components = *beam.at(beam_i);
                    forwardDecoderByOneStep(graph, decoder_components,
                            i == 0 ? BEGIN_SYMBOL : last_answers.at(beam_i), hyper_params,
                            model_params);
                    if (i == 0) {
                        break;
                    }
                }

                graph.forward();
            }
        }

        if (word_ids_result.size() < default_config.result_count_factor * k) {
            cerr << boost::format("word_ids_result size is %1%, but beam_size is %2%") %
                word_ids_result.size() % k << endl;
            abort();
        }

        for (const auto &pair : word_ids_result) {
            const vector<int> ids = pair.first;
            cout << boost::format("beam result:%1%") % exp(pair.second) << endl;
            printWordIds(ids, model_params.lookup_table);
        }

        auto compair = [](const pair<vector<int>, dtype> &a, const pair<vector<int>, dtype> &b) {
            return a.second < b.second;
        };
        auto max = max_element(word_ids_result.begin(), word_ids_result.end(), compair);

        return make_pair(max->first, exp(max->second));
    }
};

#endif
