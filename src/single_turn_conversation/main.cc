#include "cxxopts.hpp"
#include "custom_exceptions.h"
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <random>
#include "INIReader.h"
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <mutex>
#include <atomic>
#include <exception>
#include <boost/format.hpp>
#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include "N3LDG.h"
#include "single_turn_conversation/data_manager.h"
#include "single_turn_conversation/def.h"
#include "single_turn_conversation/bleu.h"
#include "single_turn_conversation/perplex.h"
#include "single_turn_conversation/default_config.h"
#include "single_turn_conversation/encoder_decoder/graph_builder.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"

using namespace std;
using namespace cxxopts;
using namespace boost::asio;
using boost::is_any_of;
using boost::format;
using boost::filesystem::path;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;

unordered_map<string, float> calculateIdf(const vector<vector<string>> sentences) {
    cout << "sentences size:" << sentences.size() << endl;
    unordered_map<string, int> doc_counts;
    int i = 0;
    for (const vector<string> &sentence : sentences) {
        if (i++ % 10000 == 0) {
            cout << i << " ";
        }
        set<string> words;
        for (const string &word : sentence) {
            words.insert(word);
        }

        for (const string &word : words) {
            auto it = doc_counts.find(word);
            if (it == doc_counts.end()) {
                doc_counts.insert(make_pair(word, 1));
            } else {
                ++doc_counts.at(word);
            }
        }
    }
    cout << endl;

    unordered_map<string, float> result;
    for (const auto &it : doc_counts) {
        float idf = log(sentences.size() / static_cast<float>(it.second));
        if (idf < 0.0) {
            cerr << "idf:" << idf << endl;
            abort();
        }

        utf8_string utf8(it.first);
        result.insert(make_pair(it.first, idf));
    }

    return result;
}

void addWord(unordered_map<string, int> &word_counts, const string &word) {
    auto it = word_counts.find(word);
    if (it == word_counts.end()) {
        word_counts.insert(make_pair(word, 1));
    } else {
        it->second++;
    }
}

void addWord(unordered_map<string, int> &word_counts, const vector<string> &sentence) {
    for (const string &word : sentence) {
        addWord(word_counts, word);
    }
}

DefaultConfig parseDefaultConfig(INIReader &ini_reader) {
    DefaultConfig default_config;
    static const string SECTION = "default";
    default_config.train_pair_file = ini_reader.Get(SECTION, "train_pair_file", "");
    if (default_config.train_pair_file.empty()) {
        cerr << "pair file empty" << endl;
        abort();
    }

    default_config.dev_pair_file = ini_reader.Get(SECTION, "dev_pair_file", "");
    if (default_config.dev_pair_file.empty()) {
        cerr << "dev file empty" << endl;
        abort();
    }

    default_config.test_pair_file = ini_reader.Get(SECTION, "test_pair_file", "");
    if (default_config.test_pair_file.empty()) {
        cerr << "test file empty" << endl;
        abort();
    }

    default_config.post_file = ini_reader.Get(SECTION, "post_file", "");
    if (default_config.post_file.empty()) {
        cerr << "post file empty" << endl;
        abort();
    }

    default_config.response_file = ini_reader.Get(SECTION, "response_file", "");
    if (default_config.post_file.empty()) {
        cerr << "response file empty" << endl;
        abort();
    }

    string program_mode_str = ini_reader.Get(SECTION, "program_mode", "");
    ProgramMode program_mode;
    if (program_mode_str == "interacting") {
        program_mode = ProgramMode::INTERACTING;
    } else if (program_mode_str == "training") {
        program_mode = ProgramMode::TRAINING;
    } else if (program_mode_str == "decoding") {
        program_mode = ProgramMode::DECODING;
    } else if (program_mode_str == "metric") {
        program_mode = ProgramMode::METRIC;
    } else {
        cout << format("program mode is %1%") % program_mode_str << endl;
        abort();
    }
    default_config.program_mode = program_mode;

    default_config.check_grad = ini_reader.GetBoolean(SECTION, "check_grad", false);
    default_config.one_response = ini_reader.GetBoolean(SECTION, "one_response", false);
    default_config.learn_test = ini_reader.GetBoolean(SECTION, "learn_test", false);
    default_config.save_model_per_batch = ini_reader.GetBoolean(SECTION, "save_model_per_batch",
            false);
    default_config.split_unknown_words = ini_reader.GetBoolean(SECTION, "split_unknown_words",
            true);

    default_config.train_sample_count = ini_reader.GetInteger(SECTION, "train_sample_count",
            1000000000);
    default_config.dev_sample_count = ini_reader.GetInteger(SECTION, "dev_sample_count",
            1000000000);
    default_config.test_sample_count = ini_reader.GetInteger(SECTION, "test_sample_count",
            1000000000);
    default_config.hold_batch_size = ini_reader.GetInteger(SECTION, "hold_batch_size", 100);
    default_config.device_id = ini_reader.GetInteger(SECTION, "device_id", 0);
    default_config.seed = ini_reader.GetInteger(SECTION, "seed", 0);
    default_config.cut_length = ini_reader.GetInteger(SECTION, "cut_length", 30);
    default_config.keyword_bound = ini_reader.GetReal(SECTION, "keyword_bound", 0);
    default_config.keyword_fork_bound = ini_reader.GetReal(SECTION, "keyword_fork_bound", 0);
    default_config.max_epoch = ini_reader.GetInteger(SECTION, "max_epoch", 100);
    default_config.output_model_file_prefix = ini_reader.Get(SECTION, "output_model_file_prefix",
            "");
    default_config.input_model_file = ini_reader.Get(SECTION, "input_model_file", "");
    default_config.input_model_dir = ini_reader.Get(SECTION, "input_model_dir", "");
    default_config.black_list_file = ini_reader.Get(SECTION, "black_list_file", "");
    default_config.memory_in_gb = ini_reader.GetReal(SECTION, "memory_in_gb", 0.0f);
    default_config.ngram_penalty_1 = ini_reader.GetReal(SECTION, "ngram_penalty_1", 0.0f);
    default_config.ngram_penalty_2 = ini_reader.GetReal(SECTION, "ngram_penalty_2", 0.0f);
    default_config.ngram_penalty_3 = ini_reader.GetReal(SECTION, "ngram_penalty_3", 0.0f);
    default_config.result_count_factor = ini_reader.GetReal(SECTION, "result_count_factor", 1.0f);

    return default_config;
}

HyperParams parseHyperParams(INIReader &ini_reader) {
    HyperParams hyper_params;

    int keyword_decoder_layer = ini_reader.GetInteger("hyper", "keyword_decoder_layer", 1);
    hyper_params.keyword_decoder_layer = keyword_decoder_layer;

    int word_dim = ini_reader.GetInteger("hyper", "word_dim", 0);
    if (word_dim <= 0) {
        cerr << "word_dim wrong" << endl;
        abort();
    }
    hyper_params.word_dim = word_dim;

    int encoding_hidden_dim = ini_reader.GetInteger("hyper", "hidden_dim", 0);
    if (encoding_hidden_dim <= 0) {
        cerr << "hidden_dim wrong" << endl;
        abort();
    }
    hyper_params.hidden_dim = encoding_hidden_dim;

    float dropout = ini_reader.GetReal("hyper", "dropout", 0.0);
    if (dropout < -1.0f || dropout >=1.0f) {
        cerr << "dropout wrong" << endl;
        abort();
    }
    hyper_params.dropout = dropout;

    int batch_size = ini_reader.GetInteger("hyper", "batch_size", 0);
    if (batch_size == 0) {
        cerr << "batch_size not found" << endl;
        abort();
    }
    hyper_params.batch_size = batch_size;

    int beam_size = ini_reader.GetInteger("hyper", "beam_size", 0);
    if (beam_size == 0) {
        cerr << "beam_size not found" << endl;
        abort();
    }
    hyper_params.beam_size = beam_size;

    float learning_rate = ini_reader.GetReal("hyper", "learning_rate", 0.001f);
    if (learning_rate <= 0.0f) {
        cerr << "learning_rate wrong" << endl;
        abort();
    }
    hyper_params.learning_rate = learning_rate;

    float min_learning_rate = ini_reader.GetReal("hyper", "min_learning_rate", 0.0001f);
    if (min_learning_rate <= 0.0f) {
        cerr << "min_learning_rate wrong" << endl;
        abort();
    }
    hyper_params.min_learning_rate = min_learning_rate;

    float learning_rate_decay = ini_reader.GetReal("hyper", "learning_rate_decay", 0.9f);
    if (learning_rate_decay <= 0.0f || learning_rate_decay > 1.0f) {
        cerr << "decay wrong" << endl;
        abort();
    }
    hyper_params.learning_rate_decay = learning_rate_decay;

    float warm_up_learning_rate = ini_reader.GetReal("hyper", "warm_up_learning_rate", 1e-6);
    if (warm_up_learning_rate < 0 || warm_up_learning_rate > 1.0f) {
        cerr << "warm_up_learning_rate wrong" << endl;
        abort();
    }
    hyper_params.warm_up_learning_rate = warm_up_learning_rate;

    int warm_up_iterations = ini_reader.GetInteger("hyper", "warm_up_iterations", 1000);
    if (warm_up_iterations < 0) {
        cerr << "warm_up_iterations wrong" << endl;
        abort();
    }
    hyper_params.warm_up_iterations = warm_up_iterations;

    int word_cutoff = ini_reader.GetReal("hyper", "word_cutoff", -1);
    if(word_cutoff == -1){
   	cerr << "word_cutoff read error" << endl;
    }
    hyper_params.word_cutoff = word_cutoff;

    bool word_finetune = ini_reader.GetBoolean("hyper", "word_finetune", -1);
    hyper_params.word_finetune = word_finetune;

    string word_file = ini_reader.Get("hyper", "word_file", "");
    hyper_params.word_file = word_file;

    float l2_reg = ini_reader.GetReal("hyper", "l2_reg", 0.0f);
    if (l2_reg < 0.0f || l2_reg > 1.0f) {
        cerr << "l2_reg:" << l2_reg << endl;
        abort();
    }
    hyper_params.l2_reg = l2_reg;
    string optimizer = ini_reader.Get("hyper", "optimzer", "");
    if (optimizer == "adam") {
        hyper_params.optimizer = Optimizer::ADAM;
    } else if (optimizer == "adagrad") {
        hyper_params.optimizer = Optimizer::ADAGRAD;
    } else if (optimizer == "adamw") {
        hyper_params.optimizer = Optimizer::ADAMW;
    } else {
        cerr << "invalid optimzer:" << optimizer << endl;
        abort();
    }

    return hyper_params;
}

vector<int> toIds(const vector<string> &sentence, const LookupTable<Param> &lookup_table,
        bool permit_unkown = true) {
    vector<int> ids;
    for (const string &word : sentence) {
	int xid = lookup_table.getElemId(word);
        if (!permit_unkown && xid == lookup_table.elems.from_string(::unknownkey)) {
            cerr << "toIds error: unknown word " << word << endl;
            abort();
        }
        if (xid >= lookup_table.nVSize) {
            cerr << "xid:" << xid << " word:" << word << endl;
            for (const string &w :sentence) {
                cerr << w;
            }
            cerr << endl;
            abort();
        }
        ids.push_back(xid);
    }
    return ids;
}

void printWordIds(const vector<int> &word_ids, const LookupTable<Param> &lookup_table,
        int offset = 0) {
    for (int word_id : word_ids) {
        cout << lookup_table.elems.from_id(word_id + offset) << " ";
    }
    cout << endl;
}

void printWordIdsWithKeywords(const vector<int> &word_ids, const LookupTable<Param> &lookup_table) {
    for (int i = 0; i < word_ids.size(); i += 2) {
        int word_id = word_ids.at(i);
        cout << lookup_table.elems.from_id(word_id) << " ";
    }
    cout << endl;
    for (int i = 1; i < word_ids.size(); i += 2) {
        int word_id = word_ids.at(i);
        cout << lookup_table.elems.from_id(word_id) << " ";
    }
    cout << endl;
}

void analyze(const vector<int> &results, const vector<int> &answers, Metric &metric) {
    if (results.size() != answers.size()) {
        cerr << "results size is not equal to answers size" << endl;
        cerr << boost::format("results size:%1% answers size:%2%\n") % results.size() %
            answers.size();
        abort();
    }

    int size = results.size();
    for (int i = 0; i < size; ++i) {
        ++metric.overall_label_count;
        if (results.at(i) == answers.at(i)) {
            ++metric.correct_label_count;
        }
    }
}

string saveModel(const HyperParams &hyper_params, ModelParams &model_params,
        const string &filename_prefix, int epoch) {
    cout << "saving model file..." << endl;
    auto t = time(nullptr);
    auto tm = *localtime(&t);
    ostringstream oss;
    oss << put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    string filename = filename_prefix + oss.str() + "-epoch" + to_string(epoch);
#if USE_GPU
    model_params.copyFromDeviceToHost();
#endif

    Json::Value root;
    root["hyper_params"] = hyper_params.toJson();
    root["model_params"] = model_params.toJson();
    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "";
    string json_str = Json::writeString(builder, root);
    ofstream out(filename);
    out << json_str;
    out.close();
    cout << format("model file %1% saved") % filename << endl;
    return filename;
}

shared_ptr<Json::Value> loadModel(const string &filename) {
    ifstream is(filename.c_str());
    shared_ptr<Json::Value> root(new Json::Value);
    if (is) {
        cout << "loading model..." << endl;
        stringstream sstr;
        sstr << is.rdbuf();
        string str = sstr.str();
        Json::CharReaderBuilder builder;
        auto reader = unique_ptr<Json::CharReader>(builder.newCharReader());
        string error;
        if (!reader->parse(str.c_str(), str.c_str() + str.size(), root.get(), &error)) {
            cerr << boost::format("parse json error:%1%") % error << endl;
            abort();
        }
        cout << "model loaded" << endl;
    } else {
        cerr << format("failed to open is, error when loading %1%") % filename << endl;
        abort();
    }

    return root;
}

void loadModel(const DefaultConfig &default_config, HyperParams &hyper_params,
        ModelParams &model_params,
        const Json::Value *root,
        const function<void(const DefaultConfig &default_config, const HyperParams &hyper_params,
            ModelParams &model_params, const Alphabet*)> &allocate_model_params) {
    hyper_params.fromJson((*root)["hyper_params"]);
    hyper_params.print();
    allocate_model_params(default_config, hyper_params, model_params, nullptr);
    model_params.fromJson((*root)["model_params"]);
#if USE_GPU
    model_params.copyFromHostToDevice();
#endif
}

pair<vector<Node *>, vector<int>> keywordNodesAndIds(const DecoderComponents &decoder_components,
        const WordIdfInfo &idf_info,
        const ModelParams &model_params) {
    vector<Node *> keyword_result_nodes = decoder_components.wordvector_to_onehots;
    vector<int> keyword_ids = toIds(idf_info.keywords_behind, model_params.lookup_table, true);
    vector<Node *> non_null_nodes;
    vector<int> chnanged_keyword_ids;
    int stop_id = model_params.lookup_table.elems.from_string(STOP_SYMBOL);
    for (int j = 0; j < keyword_result_nodes.size(); ++j) {
        if (keyword_result_nodes.at(j) != nullptr) {
            non_null_nodes.push_back(keyword_result_nodes.at(j));
            int id = keyword_ids.at(j) - stop_id;
            if (id < 0) {
                cerr << "keywordNodesAndIds - id is less than 0:" << id << endl;
                abort();
            }
            chnanged_keyword_ids.push_back(id);
        }
    }

    return {non_null_nodes, chnanged_keyword_ids};
}


float metricTestPosts(const HyperParams &hyper_params, ModelParams &model_params,
        const vector<PostAndResponses> &post_and_responses_vector,
        const vector<vector<string>> &post_sentences,
        const vector<vector<string>> &response_sentences,
        const vector<WordIdfInfo> &response_idf_info_list) {
    cout << "metricTestPosts begin" << endl;
    hyper_params.print();
    float rep_perplex(0.0f);
    thread_pool pool(16);
    mutex rep_perplex_mutex;
    int size_sum = 0;
    int keyword_size_sum_sum = 0;
    float normal_ppl_sum = 0;
    float keyword_ppl_sum = 0;

    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        auto f = [&]() {
            cout << "post:" << endl;
            auto post = post_sentences.at(post_and_responses.post_id);
            print(post);

            const vector<int> &response_ids = post_and_responses.response_ids;
            float avg_perplex = 0.0f;
            float normal_sum = 0;
            float keyword_sum = 0;
            int sum = 0;
            int keyword_size_sum = 0;
            cout << "response size:" << response_ids.size() << endl;
            for (int response_id : response_ids) {
//                cout << "response:" << endl;
//                auto response = response_sentences.at(response_id);
//                print(response);
                const WordIdfInfo &idf_info = response_idf_info_list.at(response_id);
//                print(idf_info.keywords_behind);
                n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
                profiler.BeginEvent("build computation graph");
                Graph graph;
                GraphBuilder graph_builder;
                graph_builder.forward(graph, post_sentences.at(post_and_responses.post_id),
                        hyper_params, model_params, false);
                DecoderComponents keyword_decoder(hyper_params.keyword_decoder_layer),
                                  normal_decoder(1);
                graph_builder.forwardDecoder(graph, normal_decoder, keyword_decoder,
                        response_sentences.at(response_id),
                        idf_info.keywords_behind,
                        hyper_params, model_params, false);
                profiler.EndEvent();
                graph.compute();
                vector<Node*> nodes = toNodePointers(normal_decoder.wordvector_to_onehots);
                vector<int> word_ids = transferVector<int, string>(
                        response_sentences.at(response_id), [&](const string &w) -> int {
                        return model_params.lookup_table.getElemId(w);
                        });
                int sentence_len = nodes.size();
                float normal_perplex = computePerplex(nodes, word_ids, sentence_len);
                normal_sum += normal_perplex;
                int word_ids_size = word_ids.size();
                auto keyword_nodes_and_ids = keywordNodesAndIds(keyword_decoder, idf_info,
                        model_params);
                keyword_size_sum += keyword_nodes_and_ids.first.size();
                float keyword_ppl = computePerplex(keyword_nodes_and_ids.first,
                        keyword_nodes_and_ids.second, sentence_len);
                keyword_sum += keyword_ppl;
                for (int i = 0; i < keyword_nodes_and_ids.first.size(); ++i) {
                    nodes.push_back(keyword_nodes_and_ids.first.at(i));
                    word_ids.push_back(keyword_nodes_and_ids.second.at(i));
                }

                float perplex = computePerplex(nodes, word_ids, sentence_len);
                avg_perplex += perplex;
                sum += word_ids_size;
            }
            cout << "size:" << response_ids.size() << endl;
            cout << "avg_perplex:" << exp(avg_perplex/sum) << endl;
            cout << "normal_perplex:" << exp(normal_sum/sum) << endl;
            cout << "keyword_perplex:" << exp(keyword_sum/keyword_size_sum) << endl;
            rep_perplex_mutex.lock();
            rep_perplex += avg_perplex;
            size_sum += sum;
            keyword_size_sum_sum += keyword_size_sum;
            normal_ppl_sum += normal_sum;
            keyword_ppl_sum += keyword_sum;
            rep_perplex_mutex.unlock();
        };
        post(pool, f);
    }
    pool.join();
    rep_perplex = exp(rep_perplex / size_sum);

    cout << "total avg perplex:" << rep_perplex << endl;
    cout << "keyword:" << exp(keyword_ppl_sum / keyword_size_sum_sum);
    cout << "normal:" << exp(normal_ppl_sum / size_sum);
    return rep_perplex;
}

void decodeTestPosts(const HyperParams &hyper_params, ModelParams &model_params,
        DefaultConfig &default_config,
        const unordered_map<string, float> & word_idf_table,
        const vector<WordIdfInfo> &response_idf_info_list,
        const vector<PostAndResponses> &post_and_responses_vector,
        const vector<vector<string>> &post_sentences,
        const vector<vector<string>> &response_sentences,
        const vector<string> &black_list) {
    cout << "decodeTestPosts begin" << endl;
    hyper_params.print();
    vector<CandidateAndReferences> candidate_and_references_vector;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        cout << "post:" << endl;
        auto post_sentence = post_sentences.at(post_and_responses.post_id);
        print(post_sentence);
        Graph graph;
        GraphBuilder graph_builder;
        graph_builder.forward(graph, post_sentences.at(post_and_responses.post_id),
                hyper_params, model_params, false);
        vector<DecoderComponents> decoder_components_vector;
//        decoder_components_vector.resize(hyper_params.beam_size);
        auto pair = graph_builder.forwardDecoderUsingBeamSearch(graph, decoder_components_vector,
                word_idf_table, hyper_params.beam_size, hyper_params, model_params, default_config,
                black_list);
        const vector<WordIdAndProbability> &word_ids_and_probability = pair.first;
        cout << "post:" << endl;
        print(post_sentences.at(post_and_responses.post_id));
        cout << "response:" << endl;
        printWordIdsWithKeywords(word_ids_and_probability, model_params.lookup_table,
                word_idf_table);
        dtype probability = pair.second;
        cout << format("probability:%1%") % probability << endl;
        if (word_ids_and_probability.empty()) {
            continue;
        }

        auto to_word = [&](const WordIdAndProbability &in) {
            return model_params.lookup_table.elems.from_id(in.word_id);
        };
        vector<string> decoded;
        transform(word_ids_and_probability.begin(), word_ids_and_probability.end(),
                back_inserter(decoded), to_word);
        vector<string> filtered;
        for (int i = 1; i < decoded.size(); i += 2) {
            filtered.push_back(decoded.at(i));
        }
        filtered.pop_back();

        const vector<int> &response_ids = post_and_responses.response_ids;
        vector<vector<string>> str_references =
            transferVector<vector<string>, int>(response_ids,
                    [&](int response_id) -> vector<string> {
                    return response_sentences.at(response_id);
                    });
        vector<vector<string>> id_references;
        for (const vector<string> &strs : str_references) {
            auto stop_removed = strs;
            stop_removed.pop_back();
            id_references.push_back(stop_removed);
        }

        CandidateAndReferences candidate_and_references(filtered, id_references);
        candidate_and_references_vector.push_back(candidate_and_references);

        for (int ngram = 1; ngram <=4; ++ngram) {
            float bleu_value = computeBleu(candidate_and_references_vector, ngram);
            cout << "bleu_" << ngram << ":" << bleu_value << endl;
        }
    }
}

void interact(const DefaultConfig &default_config, const HyperParams &hyper_params,
        ModelParams &model_params,
        unordered_map<string, float> &word_idfs,
        unordered_map<string, int> &word_counts,
        int word_cutoff,
        const vector<string> black_list) {
}

pair<unordered_set<int>, unordered_set<int>> PostAndResponseIds(
        const vector<PostAndResponses> &post_and_responses_vector) {
    unordered_set<int> post_ids, response_ids;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        post_ids.insert(post_and_responses.post_id);
        for (int id : post_and_responses.response_ids) {
            response_ids.insert(id);
        }
    }
    return make_pair(post_ids, response_ids);
}

unordered_set<string> knownWords(const unordered_map<string, int> &word_counts, int word_cutoff) {
    unordered_set<string> word_set;
    for (auto it : word_counts) {
        if (it.second > word_cutoff) {
            word_set.insert(it.first);
        }
    }
    return word_set;
}

unordered_set<string> knownWords(const vector<string> &words) {
    unordered_set<string> word_set;
    for (const string& w : words) {
        word_set.insert(w);
    }
    return word_set;
}

vector<string> getAllWordsByIdfAscendingly(const unordered_map<string, float> &idf_table,
        const unordered_map<string, int> &word_count_table,
        int word_cutoff) {
    vector<string> result;
    for (auto &it : word_count_table) {
        if (it.second > word_cutoff) {
            result.push_back(it.first);
        }
    }

    auto cmp = [&idf_table](const string &a, const string &b) -> bool {
        return idf_table.at(a) < idf_table.at(b);
    };

    sort(result.begin(), result.end(), cmp);

    return result;
}

std::pair<dtype, std::vector<int>> MaxLogProbabilityLossWithInconsistentDims(
        const std::vector<Node*> &result_nodes,
        const std::vector<int> &ids,
        int batchsize,
        int vocabulary_size) {
    if (ids.size() != result_nodes.size()) {
        cerr << "ids size is not equal to result_nodes'." << endl;
        abort();
    }

    pair<dtype, std::vector<int>> final_result;

    for (int i = 0; i < result_nodes.size(); ++i) {
        vector<int> id = {ids.at(i)};
        vector<Node *> node = {result_nodes.at(i)};
        if (id.front() >= node.front()->getDim()) {
            cerr << "i:" << i << endl;
            cerr << boost::format("id:%1% dim:%2%") % id.front() % node.front()->getDim() << endl;
            Json::Value info;
            info["i"] = i;
            throw InformedRuntimeError(info);
        }
        auto result = maxLogProbabilityLoss(node, id, 1.0 / batchsize);
        if (result.second.size() != 1) {
            cerr << "result second size:" << result.second.size() << endl;
            abort();
        }
        final_result.first += result.first;
        int word_id = result.second.front();
        if (word_id < 0 || word_id > vocabulary_size) {
            cerr << "word_id:" << word_id << endl;
            node.front()->getVal().print();
            cout << node.front()->getNodeIndex() << endl;
            abort();
        }
        final_result.second.push_back(word_id);
    }

    return final_result;
}

template<typename T>
void preserveVector(vector<T> &vec, int count, int seed) {
    default_random_engine engine(seed);
    shuffle(vec.begin(), vec.end(), engine);
    vec.erase(vec.begin() + std::min<int>(count, vec.size()), vec.end());
}

int main(int argc, char *argv[]) {
    cout << "dtype size:" << sizeof(dtype) << endl;

    Options options("single-turn-conversation", "single turn conversation");
    options.add_options()
        ("config", "config file name", cxxopts::value<string>());
    auto args = options.parse(argc, argv);

    string configfilename = args["config"].as<string>();
    INIReader ini_reader(configfilename);
    if (ini_reader.ParseError() < 0) {
        cerr << "parse ini failed" << endl;
        abort();
    }

    DefaultConfig &default_config = GetDefaultConfig();
    default_config = parseDefaultConfig(ini_reader);
    cout << "default_config:" << endl;
    default_config.print();

#if USE_GPU
    n3ldg_cuda::InitCuda(default_config.device_id, default_config.memory_in_gb);
#endif
    globalPoolEnabled() = true;
    globalLimitedDimEnabled() = true;

    HyperParams hyper_params = parseHyperParams(ini_reader);
    cout << "hyper_params:" << endl;
    hyper_params.print();

    vector<PostAndResponses> train_post_and_responses = readPostAndResponsesVector(
            default_config.train_pair_file);
    preserveVector(train_post_and_responses, default_config.train_sample_count,
            default_config.seed);
    cout << "train_post_and_responses_vector size:" << train_post_and_responses.size()
        << endl;
    vector<PostAndResponses> dev_post_and_responses = readPostAndResponsesVector(
            default_config.dev_pair_file);
    preserveVector(dev_post_and_responses, default_config.dev_sample_count, default_config.seed);
    cout << "dev_post_and_responses_vector size:" << dev_post_and_responses.size()
        << endl;
    vector<PostAndResponses> test_post_and_responses = readPostAndResponsesVector(
            default_config.test_pair_file);
    preserveVector(test_post_and_responses, default_config.test_sample_count, default_config.seed);
    cout << "test_post_and_responses_vector size:" << test_post_and_responses.size()
        << endl;
    vector<ConversationPair> train_conversation_pairs;
    for (const PostAndResponses &post_and_responses : train_post_and_responses) {
        vector<ConversationPair> conversation_pairs = toConversationPairs(post_and_responses);
        for (ConversationPair &conversation_pair : conversation_pairs) {
            train_conversation_pairs.push_back(move(conversation_pair));
        }
    }

    vector<vector<string>> post_sentences = readSentences(default_config.post_file);
    vector<vector<string>> response_sentences = readSentences(default_config.response_file);
    vector<bool> is_response_in_train_set;
    is_response_in_train_set.resize(response_sentences.size(), false);
    for (const auto &p : train_conversation_pairs) {
        is_response_in_train_set.at(p.response_id) = true;
    }

    Alphabet alphabet;
    shared_ptr<Json::Value> root_ptr;
    unordered_map<string, int> word_counts;
    auto wordStat = [&]() {
        for (const ConversationPair &conversation_pair : train_conversation_pairs) {
            const vector<string> &post_sentence = post_sentences.at(conversation_pair.post_id);
            addWord(word_counts, post_sentence);

            const vector<string> &response_sentence = response_sentences.at(
                    conversation_pair.response_id);
            addWord(word_counts, response_sentence);
        }

        if (hyper_params.word_file != "" && !hyper_params.word_finetune) {
            for (const PostAndResponses &dev : dev_post_and_responses){
                const vector<string>&post_sentence = post_sentences.at(dev.post_id);
                addWord(word_counts, post_sentence);

                for(int i=0; i<dev.response_ids.size(); i++){
                    const vector<string>&resp_sentence = response_sentences.at(
                            dev.response_ids.at(i));
                    addWord(word_counts, resp_sentence);
                }
            }

            for (const PostAndResponses &test : test_post_and_responses){
                const vector<string>&post_sentence = post_sentences.at(test.post_id);
                addWord(word_counts, post_sentence);

                for(int i =0; i<test.response_ids.size(); i++){
                    const vector<string>&resp_sentence =
                        response_sentences.at(test.response_ids.at(i));
                    addWord(word_counts, resp_sentence);
                }
            }
        }
    };
    wordStat();

    word_counts[unknownkey] = 1000000000;

    for (auto &s : post_sentences) {
        for (string &w : s) {
            const auto &it = word_counts.find(w);
            if (it == word_counts.end() || it->second <= hyper_params.word_cutoff) {
                w = unknownkey;
            }
        }
    }
    for (auto &s : response_sentences) {
        for (string &w : s) {
            const auto &it = word_counts.find(w);
            if (it == word_counts.end() || it->second <= hyper_params.word_cutoff) {
                w = unknownkey;
            }
        }
    }

    vector<vector<string>> all_sentences;
    for (auto &p : train_conversation_pairs) {
        auto &s = response_sentences.at(p.response_id);
        all_sentences.push_back(s);
        auto &s2 = post_sentences.at(p.post_id);
        all_sentences.push_back(s2);
    }
    cout << "merged" << endl;
    cout << "calculating idf" << endl;
    auto all_idf = calculateIdf(all_sentences);
    cout << "idf calculated" << endl;
    vector<string> all_word_list = getAllWordsByIdfAscendingly(all_idf, word_counts,
                        hyper_params.word_cutoff);
    cout << "all_word_list size:" << all_word_list.size() << endl;
    for (int i = 0; i < all_word_list.size(); ++i) {
        cout << all_word_list.at(i) << ":" ;
        cout << all_idf.at(all_word_list.at(i)) << " ";
        cout << word_counts.at(all_word_list.at(i)) << "  ";
    }
    alphabet.init(all_word_list);
    cout << boost::format("alphabet size:%1%") % alphabet.size() << endl;

    ModelParams model_params;
    int beam_size = hyper_params.beam_size;

    auto allocate_model_params = [](const DefaultConfig &default_config,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            const Alphabet *alphabet) {
        cout << format("allocate word_file:%1%\n") % hyper_params.word_file;
        if (alphabet != nullptr) {
            if(hyper_params.word_file != "" &&
                    default_config.program_mode == ProgramMode::TRAINING &&
                    default_config.input_model_file == "") {
                model_params.lookup_table.init(*alphabet, hyper_params.word_file,
                        hyper_params.word_finetune);
            } else {
                model_params.lookup_table.init(*alphabet, hyper_params.word_dim, true);
            }
        }
        model_params.attention_params.init(hyper_params.hidden_dim, hyper_params.hidden_dim);
        model_params.clip_input_params.init(hyper_params.hidden_dim,
                hyper_params.word_dim + hyper_params.hidden_dim);
        model_params.keyword_attention_params.init(hyper_params.hidden_dim,
                hyper_params.hidden_dim);
        model_params.left_to_right_encoder_params.init(hyper_params.hidden_dim,
                hyper_params.word_dim);
        model_params.left_to_right_decoder_params.init(hyper_params.hidden_dim,
                2 * hyper_params.word_dim + hyper_params.hidden_dim);
        function<void(LSTM1Params &, int)> init_param = [&](LSTM1Params &param, int layer) {
            param.init(hyper_params.hidden_dim, layer == 0 ?
                    hyper_params.word_dim + hyper_params.hidden_dim : hyper_params.hidden_dim);
        };
        model_params.keyword_decoder_params.init(hyper_params.keyword_decoder_layer, init_param);
        model_params.hidden_to_wordvector_params.init(hyper_params.word_dim,
                2 * hyper_params.hidden_dim + 2 * hyper_params.word_dim, false);
        model_params.hidden_to_keyword_params.init(hyper_params.word_dim,
                2 * hyper_params.hidden_dim + hyper_params.word_dim, false);
    };

    if (default_config.program_mode != ProgramMode::METRIC) {
        if (default_config.input_model_file == "") {
            allocate_model_params(default_config, hyper_params, model_params, &alphabet);
            cout << "complete allocate" << endl;
        } else {
            root_ptr = loadModel(default_config.input_model_file);
            loadModel(default_config, hyper_params, model_params, root_ptr.get(),
                    allocate_model_params);
        }
    } else {
        if (default_config.input_model_file == "") {
            abort();
        } else {
            root_ptr = loadModel(default_config.input_model_file);
            loadModel(default_config, hyper_params, model_params, root_ptr.get(),
                    allocate_model_params);
        }
    }
    auto black_list = readBlackList(default_config.black_list_file);

    cout << "completed" << endl;
    cout << "reading response idf info ..." << endl;
    vector<WordIdfInfo> response_idf_info_list = readWordIdfInfoList(response_sentences,
            is_response_in_train_set, all_idf, word_counts,
            model_params.lookup_table.elems.m_string_to_id, hyper_params.word_cutoff);
    cout << "completed" << endl;

    if (default_config.program_mode == ProgramMode::INTERACTING) {
        hyper_params.beam_size = beam_size;
        interact(default_config, hyper_params, model_params, all_idf, word_counts,
                hyper_params.word_cutoff, black_list);
    } else if (default_config.program_mode == ProgramMode::DECODING) {
        hyper_params.beam_size = beam_size;
        decodeTestPosts(hyper_params, model_params, default_config, all_idf,
                response_idf_info_list, test_post_and_responses, post_sentences,
                response_sentences, black_list);
    } else if (default_config.program_mode == ProgramMode::METRIC) {
        path dir_path(default_config.input_model_dir);
        if (!is_directory(dir_path)) {
            cerr << format("%1% is not dir path") % default_config.input_model_dir << endl;
            abort();
        }

        vector<string> ordered_file_paths;
        for(auto& entry : boost::make_iterator_range(directory_iterator(dir_path), {})) {
            string basic_name = entry.path().filename().string();
            cout << format("basic_name:%1%") % basic_name << endl;
            if (basic_name.find("model") != 0) {
                continue;
            }

            string model_file_path = entry.path().string();
            ordered_file_paths.push_back(model_file_path);
        }
        std::sort(ordered_file_paths.begin(), ordered_file_paths.end(),
                [](const string &a, const string &b)->bool {
                using boost::filesystem::last_write_time;
                return last_write_time(a) < last_write_time(b);
                });

        float max_rep_perplex = 0.0f;
        for(const string &model_file_path : ordered_file_paths) {
            cout << format("model_file_path:%1%") % model_file_path << endl;
            ModelParams model_params;
            shared_ptr<Json::Value> root_ptr = loadModel(model_file_path);
            loadModel(default_config, hyper_params, model_params, root_ptr.get(),
                    allocate_model_params);
            float rep_perplex = metricTestPosts(hyper_params, model_params, dev_post_and_responses,
                    post_sentences, response_sentences, response_idf_info_list);
            cout << format("model %1% rep_perplex is %2%") % model_file_path % rep_perplex << endl;
            if (max_rep_perplex < rep_perplex) {
                max_rep_perplex = rep_perplex;
                cout << format("best model now is %1%, and rep_perplex is %2%") % model_file_path %
                    rep_perplex << endl;
            }
        }
    } else if (default_config.program_mode == ProgramMode::TRAINING) {
        ModelUpdate model_update;
        model_update._alpha = hyper_params.learning_rate;
        model_update._reg = hyper_params.l2_reg;
        model_update.setParams(model_params.tunableParams());

        CheckGrad grad_checker;
        if (default_config.check_grad) {
            grad_checker.init(model_params.tunableParams());
        }

        dtype last_loss_sum = 1e10f;

        int iteration = 0;
        string last_saved_model;

        default_random_engine engine(default_config.seed);
        for (int epoch = 0; epoch < default_config.max_epoch; ++epoch) {
            dtype loss_sum = 0.0f;
            dtype normal_loss_sum = 0.0f;
            dtype keyword_loss_sum = 0.0f;
            cout << "epoch:" << epoch << endl;
            model_params.lookup_table.E.is_fixed = false;
            auto cmp = [&] (const ConversationPair &a, const ConversationPair &b)->bool {
                auto len = [&] (const ConversationPair &pair)->int {
                    return post_sentences.at(pair.post_id).size() +
                        response_sentences.at(pair.response_id).size();
                };
                return len(a) < len(b);
            };
            sort(begin(train_conversation_pairs), end(train_conversation_pairs), cmp);
            int valid_len = train_conversation_pairs.size() / hyper_params.batch_size *
                hyper_params.batch_size;
            int batch_count = valid_len / hyper_params.batch_size;
            cout << boost::format("valid_len:%1% batch_count:%2%") % valid_len % batch_count <<
                endl;
            for (int i = 0; i < hyper_params.batch_size; ++i) {
                auto begin_pos = begin(train_conversation_pairs) + i * batch_count;
                shuffle(begin_pos, begin_pos + batch_count, engine);
            }
            if (train_conversation_pairs.size() > hyper_params.batch_size * batch_count) {
                shuffle(begin(train_conversation_pairs) + hyper_params.batch_size * batch_count,
                        train_conversation_pairs.end(), engine);
            }


            unique_ptr<Metric> metric = unique_ptr<Metric>(new Metric);
            unique_ptr<Metric> keyword_metric = unique_ptr<Metric>(new Metric);
            n3ldg_cuda::Profiler::Reset();
            n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
            profiler.SetEnabled(false);
            profiler.BeginEvent("total");

            using namespace std::chrono;
            int duration_count = 1e3;

            for (int batch_i = 0; batch_i < batch_count +
                    (train_conversation_pairs.size() > hyper_params.batch_size * batch_count);
                    ++batch_i) {
                auto start = high_resolution_clock::now();
                cout << format("batch_i:%1% iteration:%2%") % batch_i % iteration << endl;
                if (epoch == 0) {
                    if (iteration < hyper_params.warm_up_iterations) {
                        model_update._alpha = hyper_params.warm_up_learning_rate;
                    } else {
                        model_update._alpha = hyper_params.learning_rate;
                        cout << "warm up finished, learning rate now:" <<
                            hyper_params.learning_rate << endl;
                    }
                }
                int batch_size = batch_i == batch_count ?
                    train_conversation_pairs.size() % hyper_params.batch_size :
                    hyper_params.batch_size;
                profiler.BeginEvent("build braph");
                Graph graph(false);
                vector<shared_ptr<GraphBuilder>> graph_builders;
                vector<ConversationPair> conversation_pair_in_batch;
                auto getSentenceIndex = [batch_i, batch_count](int i) {
                    return i * batch_count + batch_i;
                };
                vector<int> post_lens_in_batch;
                for (int i = 0; i < batch_size; ++i) {
                    int instance_index = getSentenceIndex(i);
                    int post_id = train_conversation_pairs.at(instance_index).post_id;
                    const auto &post_sentence = post_sentences.at(post_id);
                    post_lens_in_batch.push_back(post_sentence.size());
                }

                int max_len = *max_element(post_lens_in_batch.begin(), post_lens_in_batch.end());
                cout << "max post len:" << max_len << endl;
                Node *bucket = n3ldg_plus::bucket(graph, hyper_params.hidden_dim, 0);
                for (int pos_i = 0; pos_i < max_len; ++pos_i) {
                    for (int i = 0; i < batch_size; ++i) {
                        int instance_index = getSentenceIndex(i);
                        int post_id = train_conversation_pairs.at(instance_index).post_id;
                        const auto &post_sentence = post_sentences.at(post_id);
                        if (post_sentence.size() <= pos_i) {
                            continue;
                        }
                        shared_ptr<GraphBuilder> graph_builder(new GraphBuilder);
                        graph_builders.push_back(graph_builder);
                        conversation_pair_in_batch.push_back(train_conversation_pairs.at(
                                    instance_index));
                        graph_builder->forward(graph, post_sentence.at(pos_i), *bucket,
                                hyper_params, model_params, true);
                    }
                    graph.compute();
                }

                int response_size_sum = 0;
                vector<int> response_lens_in_batch;
                for (int i = 0; i < batch_size; ++i) {
                    int instance_index = getSentenceIndex(i);
                    int response_id = train_conversation_pairs.at(instance_index).response_id;
                    const auto& response_sentence = response_sentences.at(response_id);
                    response_size_sum += response_sentence.size();
                    response_lens_in_batch.push_back(response_sentence.size());
                }

                int max_response_len = *max_element(response_lens_in_batch.begin(),
                        response_lens_in_batch.end());

                vector<shared_ptr<DecoderComponents>> normal_decoder_vector,
                    keyword_decoder_vector;
                int keyword_size_sum = 0;
                for (int pos_i = 0; pos_i < max_response_len; ++pos_i) {
                    for (int i = 0; i < batch_size; ++i) {
                        GraphBuilder *graph_builder = graph_builders.at(i).get();
                        int instance_index = getSentenceIndex(i);
                        int response_id = train_conversation_pairs.at(instance_index).response_id;
                        if (response_sentences.at(response_id).size() <= pos_i) {
                            continue;
                        }
                        DecoderComponents *normal_decoder_ptr, *keyword_decoder_ptr;
                        if (pos_i == 0) {
                            shared_ptr<DecoderComponents> keyword_decoder(
                                    new DecoderComponents(hyper_params.keyword_decoder_layer));
                            keyword_decoder_vector.push_back(keyword_decoder);
                            shared_ptr<DecoderComponents> normal_decoder(new DecoderComponents(1));
                            normal_decoder_vector.push_back(normal_decoder);
                            normal_decoder_ptr = normal_decoder.get();
                            keyword_decoder_ptr = keyword_decoder.get();
                        } else {
                            normal_decoder_ptr = normal_decoder_vector.at(i).get();
                            keyword_decoder_ptr = keyword_decoder_vector.at(i).get();
                        }
                        const WordIdfInfo &idf_info = response_idf_info_list.at(response_id);
                        graph_builder->forwardDecoder(graph, *normal_decoder_ptr,
                                *keyword_decoder_ptr,
                                pos_i == 0 ? nullptr :
                                &response_sentences.at(response_id).at(pos_i - 1),
                                idf_info.keywords_behind.at(pos_i),
                                pos_i == 0 ? nullptr : &idf_info.keywords_behind.at(pos_i - 1),
                                pos_i, hyper_params, model_params, true);
                        keyword_size_sum += (pos_i == 0 || idf_info.keywords_behind.at(pos_i - 1)
                            == response_sentences.at(response_id).at(pos_i - 1) ? 1 : 0);
                    }
                    graph.compute();
                }

                for (int i = 0; i < batch_size; ++i) {
                    int instance_index = getSentenceIndex(i);
                    int response_id = train_conversation_pairs.at(instance_index).response_id;
                    auto response_sentence = response_sentences.at(response_id);
                    vector<int> word_ids = toIds(response_sentence, model_params.lookup_table);
                    vector<Node*> result_nodes =
                        toNodePointers(normal_decoder_vector.at(i)->wordvector_to_onehots);
                    std::pair<dtype, std::vector<int>> result;
                    try {
                        result = MaxLogProbabilityLossWithInconsistentDims(result_nodes,
                                word_ids, response_size_sum, model_params.lookup_table.nVSize);
                    } catch (const InformedRuntimeError &e) {
                        cerr << e.what() << endl;
                        int i = e.getInfo()["i"].asInt();
                        string word = response_sentence.at(i);
                        string keyword = response_idf_info_list.at(response_id).keywords_behind.at(
                                i);
                        cerr << boost::format("word idf:%1% keyword idf:%2%") %
                            all_idf.find(word)->second % all_idf.find(keyword)->second << endl;
                        print(response_idf_info_list.at(response_id).keywords_behind);
                        cerr << boost::format("%1% id:%3% %2% id:%4%") % word % keyword %
                            model_params.lookup_table.elems.from_string(word) %
                            model_params.lookup_table.elems.from_string(keyword) << endl;

                        print(response_sentence);
                        print(response_idf_info_list.at(response_id).keywords_behind);
                        abort();
                    }
                    profiler.EndCudaEvent();
                    loss_sum += result.first;
                    normal_loss_sum += result.first * response_size_sum;
                    analyze(result.second, word_ids, *metric);
                    const WordIdfInfo &response_idf = response_idf_info_list.at(response_id);
                    auto keyword_nodes_and_ids = keywordNodesAndIds(*keyword_decoder_vector.at(i),
                            response_idf, model_params);
                    profiler.BeginEvent("loss");
                    auto keyword_result = MaxLogProbabilityLossWithInconsistentDims(
                            keyword_nodes_and_ids.first, keyword_nodes_and_ids.second,
                            keyword_size_sum, model_params.lookup_table.nVSize);
                    profiler.EndCudaEvent();
                    loss_sum += keyword_result.first;
                    keyword_loss_sum += keyword_result.first * keyword_size_sum;
                    analyze(keyword_result.second, keyword_nodes_and_ids.second, *keyword_metric);

                    static int count_for_print;
                    if (++count_for_print % 100 == 1) {
                        int post_id = train_conversation_pairs.at(instance_index).post_id;
                        cout << "post:" << post_id << endl;
                        print(post_sentences.at(post_id));

                        cout << "golden answer:" << endl;
                        printWordIds(word_ids, model_params.lookup_table);
                        cout << "output:" << endl;
                        printWordIds(result.second, model_params.lookup_table);

                        int stop_id = model_params.lookup_table.elems.from_string(STOP_SYMBOL);
                        cout << "golden keywords:" << endl;
                        printWordIds(keyword_nodes_and_ids.second, model_params.lookup_table,
                                stop_id);
                        cout << "output:" << endl;
                        printWordIds(keyword_result.second, model_params.lookup_table, stop_id);
                    }
                }

                cout << "loss:" << loss_sum << "normal:" << normal_loss_sum << "keyword:" <<
                    keyword_loss_sum << endl;
                cout << "normal:" << endl;
                metric->print();
                cout << "keyword:" << endl;
                keyword_metric->print();

                graph.backward();

                if (default_config.check_grad) {
                    auto loss_function = [&](const ConversationPair &conversation_pair) -> dtype {
                        GraphBuilder graph_builder;
                        Graph graph;

                        graph_builder.forward(graph, post_sentences.at(conversation_pair.post_id),
                                hyper_params, model_params, true);

                        DecoderComponents normal_decoder(1),
                                          keyword_decoder(hyper_params.keyword_decoder_layer);
                        graph_builder.forwardDecoder(graph, normal_decoder, keyword_decoder,
                                response_sentences.at(conversation_pair.response_id),
                                response_idf_info_list.at(
                                    conversation_pair.response_id).keywords_behind,
                                hyper_params, model_params, true);

                        graph.compute();

                        vector<int> word_ids = toIds(response_sentences.at(
                                    conversation_pair.response_id), model_params.lookup_table);
                        vector<Node*> result_nodes = toNodePointers(
                                normal_decoder.wordvector_to_onehots);
                        const WordIdfInfo &response_idf = response_idf_info_list.at(
                                conversation_pair.response_id);
                        auto keyword_nodes_and_ids = keywordNodesAndIds(keyword_decoder,
                                response_idf, model_params);
                        return MaxLogProbabilityLossWithInconsistentDims(
                                keyword_nodes_and_ids.first, keyword_nodes_and_ids.second, 1,
                                model_params.lookup_table.nVSize).first +
                            MaxLogProbabilityLossWithInconsistentDims( result_nodes, word_ids, 1,
                                    model_params.lookup_table.nVSize).first;
                    };
                    cout << format("checking grad - conversation_pair size:%1%") %
                        conversation_pair_in_batch.size() << endl;
                    grad_checker.check<ConversationPair>(loss_function, conversation_pair_in_batch,
                            "");
                }

                if (hyper_params.optimizer == Optimizer::ADAM) {
                    model_update.updateAdam(10.0f);
                } else if (hyper_params.optimizer == Optimizer::ADAGRAD) {
                    model_update.update(10.0f);
                } else if (hyper_params.optimizer == Optimizer::ADAMW) {
                    model_update.updateAdamW(10.0f);
                } else {
                    cerr << "no optimzer set" << endl;
                    abort();
                }
                auto stop = high_resolution_clock::now();
                auto duration = duration_cast<milliseconds>(stop - start);
                duration_count = 0.99 * duration_count + 0.01 * duration.count();
                cout << "duration:" << duration_count << endl;

                if (default_config.save_model_per_batch) {
                    saveModel(hyper_params, model_params, default_config.output_model_file_prefix,
                            epoch);
                }

                ++iteration;
            }
            profiler.EndCudaEvent();
            profiler.Print();

            cout << "loss_sum:" << loss_sum << " last_loss_sum:" << last_loss_sum << endl;
            if (loss_sum > last_loss_sum) {
                if (epoch == 0) {
                    cerr << "loss is larger than last epoch but epoch is 0" << endl;
                    abort();
                }
                model_update._alpha *= 0.1f;
                hyper_params.learning_rate = model_update._alpha;
                cout << "learning_rate decay:" << model_update._alpha << endl;
                std::shared_ptr<Json::Value> root = loadModel(last_saved_model);
                model_params.fromJson((*root)["model_params"]);
#if USE_GPU
                model_params.copyFromHostToDevice();
#endif
            } else {
                model_update._alpha = (model_update._alpha - hyper_params.min_learning_rate) *
                    hyper_params.learning_rate_decay + hyper_params.min_learning_rate;
                hyper_params.learning_rate = model_update._alpha;
                cout << "learning_rate now:" << hyper_params.learning_rate << endl;
                last_saved_model = saveModel(hyper_params, model_params,
                        default_config.output_model_file_prefix, epoch);
            }

            last_loss_sum = loss_sum;
        }
    } else {
        abort();
    }

    return 0;
}
