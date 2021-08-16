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
#include "insnet/insnet.h"
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
        float idf = it.first == insnet::UNKNOWN_WORD ? 1e-3 :
            log(sentences.size() / static_cast<float>(it.second));
        if (idf < 0.0) {
            cerr << "idf:" << idf << endl;
            abort();
        }

        utf8_string utf8(it.first);
//        if (includePunctuation(utf8.cpp_str()) || !isPureChinese(it.first) || idf <= 5) {
//            idf = -idf;
//        }

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
    if (min_learning_rate < 0.0f) {
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
        hyper_params.optimizer = ::Optimizer::ADAM;
    } else if (optimizer == "adagrad") {
        hyper_params.optimizer = ::Optimizer::ADAGRAD;
    } else if (optimizer == "adamw") {
        hyper_params.optimizer = ::Optimizer::ADAMW;
    } else {
        cerr << "invalid optimzer:" << optimizer << endl;
        abort();
    }

    return hyper_params;
}

vector<int> toIds(const vector<string> &sentence, const Embedding<Param> &lookup_table,
        bool permit_unkown = true) {
    vector<int> ids;
    for (const string &word : sentence) {
	int xid = lookup_table.getElemId(word);
        if (!permit_unkown && xid == lookup_table.vocab.from_string(insnet::UNKNOWN_WORD)) {
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

void printWordIds(const vector<int> &word_ids, const Embedding<Param> &lookup_table) {
    for (int word_id : word_ids) {
        cout << lookup_table.vocab.from_id(word_id) << " ";
    }
    cout << endl;
}

void printWordIdsWithKeywords(const vector<int> &word_ids, const Embedding<Param> &lookup_table) {
    for (int i = 0; i < word_ids.size(); i += 2) {
        int word_id = word_ids.at(i);
        cout << lookup_table.vocab.from_id(word_id) << " ";
    }
    cout << endl;
    for (int i = 1; i < word_ids.size(); i += 2) {
        int word_id = word_ids.at(i);
        cout << lookup_table.vocab.from_id(word_id) << " ";
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

    ofstream out(filename, ios::binary);
    cereal::BinaryOutputArchive output_ar(out);
    output_ar(hyper_params, model_params, epoch);
    out.close();
    cout << format("model file %1% saved") % filename << endl;
    return filename;
}

void loadModel(const DefaultConfig &default_config, HyperParams &hyper_params,
        ModelParams &model_params,
        int &epoch,
        const string &filename,
        const function<void(const DefaultConfig &default_config, const HyperParams &hyper_params,
            ModelParams &model_params, const Vocab*)> &allocate_model_params) {
    ifstream is(filename.c_str());
    if (is) {
        cout << "loading model..." << endl;
        cereal::BinaryInputArchive ar(is);
        ar(hyper_params);
        hyper_params.print();
        allocate_model_params(default_config, hyper_params, model_params, nullptr);
        ar(model_params, epoch);
#if USE_GPU
        model_params.copyFromHostToDevice();
#endif
        cout << "model loaded" << endl;
    } else {
        cerr << fmt::format("load model fail - filename:%1%", filename) << endl;
        abort();
    }
}

pair<vector<Node *>, vector<int>> keywordNodesAndIds(const vector<Node *> &keyword_probs,
        const WordIdfInfo &idf_info,
        const ModelParams &model_params) {
    vector<int> keyword_ids = toIds(idf_info.keywords_behind, model_params.lookup_table, true);
    vector<Node *> non_null_nodes;
    vector<int> chnanged_keyword_ids;
    for (int j = 0; j < keyword_probs.size(); ++j) {
        if (keyword_probs.at(j) != nullptr) {
            non_null_nodes.push_back(keyword_probs.at(j));
            chnanged_keyword_ids.push_back(keyword_ids.at(j));
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
    float overall_ppl(0.0f), keyword_ppl_sum(0.0f), normal_ppl_sum(0.0f);
    int corpus_hit_count = 0;
    int size_sum = 0;
    int total_keyword_size_sum = 0;
    vector<int> corpus_keyword_hit_counts, corpus_keyword_sizes;
    vector<int> corpus_token_hit_counts, corpus_token_sizes;
    vector<int> corpus_unified_hit_counts, corpus_unified_sizes;

    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        cout << "post:" << endl;
        auto post = post_sentences.at(post_and_responses.post_id);
        print(post);

        const vector<int> &response_ids = post_and_responses.response_ids;
        float post_ppl = 0.0f;
        int post_hit_count = 0;
        int sum = 0;
        int keyword_sum = 0;
        vector<int> post_keyword_hit_counts, post_keyword_sizes;
        vector<int> post_token_hit_counts, post_token_sizes;
        vector<int> post_unified_hit_counts, post_unified_sizes;

        cout << "response size:" << response_ids.size() << endl;
        for (int response_id : response_ids) {
            print(response_sentences.at(response_id));
            const WordIdfInfo &idf_info = response_idf_info_list.at(response_id);
            Graph graph(ModelStage::INFERENCE);
            GraphBuilder graph_builder;
            graph_builder.forward(graph, post_sentences.at(post_and_responses.post_id),
                    hyper_params, model_params);
            auto p = graph_builder.forwardDecoder(graph, *graph_builder.encoder_hiddens,
                    response_sentences.at(response_id),
                    idf_info.keywords_behind,
                    hyper_params, model_params);
            graph.forward();
            auto nodes = p.second;
            vector<int> word_ids = transferVector<int, string>(
                    response_sentences.at(response_id), [&](const string &w) -> int {
                    return model_params.lookup_table.getElemId(w);
                    });
            for (int i = 0; i < word_ids.size(); ++i) {
                if (i != word_ids.size() - 1) {
                    --word_ids.at(i);
                } else if (word_ids.at(i) != 0) {
                    cerr << "last word id is not 0:" << word_ids.at(i) << endl;
                    abort();
                }
            }
            int word_ids_size = word_ids.size();
            auto keyword_nodes_and_ids = keywordNodesAndIds(p.first, idf_info,
                    model_params);
            int sentence_len = nodes.size();
            for (int i = 0; i < keyword_nodes_and_ids.first.size(); ++i) {
                nodes.push_back(keyword_nodes_and_ids.first.at(i));
                word_ids.push_back(keyword_nodes_and_ids.second.at(i));
            }

            int hit_count;
            vector<int> keyword_hit_flags, token_hit_flags, unified_hit_flags;
            dtype keyword_ppl, normal_ppl;
            float perplex = computePerplex(nodes, word_ids, sentence_len, hit_count,
                    keyword_hit_flags, token_hit_flags, unified_hit_flags, keyword_ppl,
                    normal_ppl);
            keyword_ppl_sum += keyword_ppl;
            normal_ppl_sum += normal_ppl;

            post_ppl += perplex;
            post_hit_count += hit_count;
            sum += word_ids_size;
            keyword_sum += nodes.size() - sentence_len;

            for(int i = 0; i < keyword_hit_flags.size(); ++i) {
                if (post_keyword_hit_counts.size() <= i) {
                    post_keyword_hit_counts.push_back(0);
                }
                post_keyword_hit_counts.at(i) += keyword_hit_flags.at(i);

                if (post_keyword_sizes.size() <= i) {
                    post_keyword_sizes.push_back(0);
                }
                post_keyword_sizes.at(i)++;
            }

            for (int i = 0; i < token_hit_flags.size(); ++i) {
                if (post_token_hit_counts.size() <= i) {
                    post_token_hit_counts.push_back(0);
                }
                post_token_hit_counts.at(i) += token_hit_flags.at(i);

                if (post_token_sizes.size() <= i) {
                    post_token_sizes.push_back(0);
                }
                post_token_sizes.at(i)++;
            }

            for (int i = 0; i < unified_hit_flags.size(); ++i) {
                if (post_unified_hit_counts.size() <= i) {
                    post_unified_hit_counts.push_back(0);
                }
                post_unified_hit_counts.at(i) += unified_hit_flags.at(i);

                if (post_unified_sizes.size() <= i) {
                    post_unified_sizes.push_back(0);
                }
                post_unified_sizes.at(i)++;
            }
        }
        cout << "size:" << response_ids.size() << endl;
        cout << "post_ppl:" << exp(post_ppl/sum) << endl;
        cout << "hit rate:" << static_cast<float>(post_hit_count) / sum << endl;
        for (int i = 0; i < post_keyword_sizes.size(); ++i) {
            cout << boost::format("keyword %1% hit rate:%2%") % i %
                (static_cast<float>(post_keyword_hit_counts.at(i)) /
                 post_keyword_sizes.at(i)) << endl;
        }
        for (int i = 0; i < post_token_sizes.size(); ++i) {
            cout << boost::format("token %1% hit rate:%2%") % i %
                (static_cast<float>(post_token_hit_counts.at(i)) /
                 post_token_sizes.at(i)) << endl;
        }
        for (int i = 0; i < post_unified_sizes.size(); ++i) {
            cout << boost::format("unified %1% hit rate:%2%") % i %
                (static_cast<float>(post_unified_hit_counts.at(i)) /
                 post_unified_sizes.at(i)) << endl;
        }
        overall_ppl += post_ppl;
        corpus_hit_count += post_hit_count;
        size_sum += sum;
        total_keyword_size_sum += keyword_sum;

        for (int i = 0; i < post_keyword_sizes.size(); ++i) {
            if (corpus_keyword_hit_counts.size() <= i) {
                corpus_keyword_hit_counts.push_back(0);
            }
            corpus_keyword_hit_counts.at(i) += post_keyword_hit_counts.at(i);

            if (corpus_keyword_sizes.size() <=i) {
                corpus_keyword_sizes.push_back(0);
            }
            corpus_keyword_sizes.at(i) += post_keyword_sizes.at(i);
        }

        for (int i = 0; i < post_token_sizes.size(); ++i) {
            if (corpus_token_hit_counts.size() <= i) {
                corpus_token_hit_counts.push_back(0);
            }
            corpus_token_hit_counts.at(i) += post_token_hit_counts.at(i);

            if (corpus_token_sizes.size() <=i) {
                corpus_token_sizes.push_back(0);
            }
            corpus_token_sizes.at(i) += post_token_sizes.at(i);
        }

        for (int i = 0; i < post_unified_sizes.size(); ++i) {
            if (corpus_unified_hit_counts.size() <= i) {
                corpus_unified_hit_counts.push_back(0);
            }
            corpus_unified_hit_counts.at(i) += post_unified_hit_counts.at(i);

            if (corpus_unified_sizes.size() <=i) {
                corpus_unified_sizes.push_back(0);
            }
            corpus_unified_sizes.at(i) += post_unified_sizes.at(i);
        }
    }
    overall_ppl = exp(overall_ppl / size_sum);
    keyword_ppl_sum = exp(keyword_ppl_sum / total_keyword_size_sum);
    normal_ppl_sum = exp(normal_ppl_sum / size_sum);
    cout << "total keyword ppl:" << keyword_ppl_sum << endl;
    cout << "total normal ppl:" << normal_ppl_sum << endl;

    cout << "total avg perplex:" << overall_ppl << endl;
    cout << "corpus hypo ppl:" << static_cast<float>(corpus_hit_count) / size_sum << endl;
    for (int i = 0; i < corpus_keyword_sizes.size(); ++i) {
        cout << boost::format("keyword %1% hit:%2% amount:%3% rate:%4%") % i %
            corpus_keyword_hit_counts.at(i) % corpus_keyword_sizes.at(i) %
            (static_cast<float>(corpus_keyword_hit_counts.at(i)) / corpus_keyword_sizes.at(i))
            << endl;
    }
    for (int i = 0; i < corpus_token_sizes.size(); ++i) {
        cout << boost::format("token %1% hit:%2% amount:%3% rate:%4%") % i %
            corpus_token_hit_counts.at(i) % corpus_token_sizes.at(i) %
            (static_cast<float>(corpus_token_hit_counts.at(i)) / corpus_token_sizes.at(i))
            << endl;
    }
    for (int i = 0; i < corpus_unified_sizes.size(); ++i) {
        cout << boost::format("unified %1% hit:%2% amount:%3% rate:%4%") % i %
            corpus_unified_hit_counts.at(i) % corpus_unified_sizes.at(i) %
            (static_cast<float>(corpus_unified_hit_counts.at(i)) / corpus_unified_sizes.at(i))
            << endl;
    }
    return overall_ppl;
}

void computeMeanAndStandardDeviation(const vector<float> &nums, float &mean, float &sd) {
    float sum = 0;
    for (float num : nums) {
        sum += num;
    }
    mean = sum / nums.size();
    if (nums.size() == 1) {
        sd = 0;
    } else {
        float variance = 0;
        for (float num : nums) {
            float x = num - mean;
            variance += x * x;
        }
        variance /= (nums.size() - 1);
        sd = sqrt(variance);
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
        dtype av, bv;
        try {
            av = idf_table.at(a);
        } catch (const std::exception &e) {
            cerr << fmt::format("getAllWordsByIdfAscendingly - {} not found", a) << endl;
            abort();
        }
        try {
            bv = idf_table.at(b);
        } catch (const std::exception &e) {
            cerr << fmt::format("getAllWordsByIdfAscendingly - {} not found", b) << endl;
            abort();
        }
        return av < bv;
    };

    sort(result.begin(), result.end(), cmp);

    return result;
}

dtype MaxLogProbabilityLossWithInconsistentDims(
        const std::vector<Node*> &result_nodes,
        const std::vector<int> &ids,
        int vocabulary_size) {
    if (ids.size() != result_nodes.size()) {
        cerr << "ids size is not equal to result_nodes'." << endl;
        abort();
    }

    dtype sum = 0;

    for (int i = 0; i < result_nodes.size(); ++i) {
        vector<int> id = {ids.at(i)};
        vector<Node *> node = {result_nodes.at(i)};
        if (id.front() >= node.front()->size()) {
            cerr << "i:" << i << endl;
            cerr << boost::format("id:%1% dim:%2%") % id.front() % node.front()->size() << endl;
            Json::Value info;
            info["i"] = i;
            throw InformedRuntimeError(info);
        }
        dtype result = insnet::NLLLoss(node, node.front()->size(), {id}, 1.0);
        sum += result;
    }

    return sum;
}

template<typename T>
void preserveVector(vector<T> &vec, int count, int seed) {
    default_random_engine engine(seed);
    shuffle(vec.begin(), vec.end(), engine);
    vec.erase(vec.begin() + std::min<int>(count, vec.size()), vec.end());
}

int main(int argc, char *argv[]) {
    globalLimitedDimEnabled() = true;
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
    cuda::initCuda(default_config.device_id, default_config.memory_in_gb);
#endif

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

    Vocab alphabet;
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

    word_counts[insnet::UNKNOWN_WORD] = 1000000000;

    for (auto &s : post_sentences) {
        for (string &w : s) {
            const auto &it = word_counts.find(w);
            if (it == word_counts.end() || it->second <= hyper_params.word_cutoff) {
                w = insnet::UNKNOWN_WORD;
            }
        }
    }
    for (auto &s : response_sentences) {
        for (string &w : s) {
            const auto &it = word_counts.find(w);
            if (it == word_counts.end() || it->second <= hyper_params.word_cutoff) {
                w = insnet::UNKNOWN_WORD;
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
    all_idf[insnet::UNKNOWN_WORD] = 0.01;
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
            const Vocab *alphabet) {
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
        model_params.attention_params.init(2 * hyper_params.hidden_dim, hyper_params.hidden_dim);
        model_params.l2r_encoder_params.init(hyper_params.hidden_dim, hyper_params.word_dim);
        model_params.r2l_encoder_params.init(hyper_params.hidden_dim, hyper_params.word_dim);
        model_params.decoder_params.init(hyper_params.hidden_dim, 2 * hyper_params.word_dim +
                2 * hyper_params.hidden_dim);
        model_params.hidden_to_wordvector_params_a.init(hyper_params.hidden_dim * 4,
                hyper_params.hidden_dim * 3 + hyper_params.word_dim * 2);
        model_params.hidden_to_wordvector_params_b.init(hyper_params.word_dim,
                hyper_params.hidden_dim * 4);
        model_params.hidden_to_keyword_params.init(hyper_params.word_dim,
                3 * hyper_params.hidden_dim, true);
    };

    int epoch_begin = -1;

    if (default_config.program_mode != ProgramMode::METRIC) {
        if (default_config.input_model_file == "") {
            allocate_model_params(default_config, hyper_params, model_params, &alphabet);
            cout << "complete allocate" << endl;
        } else {
            loadModel(default_config, hyper_params, model_params, epoch_begin,
                    default_config.input_model_file, allocate_model_params);
        }
    } else {
        globalPoolEnabled() = false;
        globalLimitedDimEnabled() = false;
        if (default_config.input_model_file == "") {
            abort();
        } else {
            loadModel(default_config, hyper_params, model_params, epoch_begin,
                    default_config.input_model_file, allocate_model_params);
            hyper_params.learning_rate_decay = ini_reader.GetFloat("hyper", "learning_rate_decay",
                    0);
            hyper_params.min_learning_rate = ini_reader.GetFloat("hyper", "min_learning_rate",
                    0);
            hyper_params.learning_rate = ini_reader.GetFloat("hyper", "learning_rate",
                    0);
            hyper_params.batch_size = ini_reader.GetFloat("hyper", "batch_size", 1);
            hyper_params.print();
        }
    }
    auto black_list = readBlackList(default_config.black_list_file);

    cout << "completed" << endl;
    cout << "reading response idf info ..." << endl;
    vector<WordIdfInfo> response_idf_info_list = readWordIdfInfoList(response_sentences,
            is_response_in_train_set, all_idf, word_counts,
            model_params.lookup_table.vocab.m_string_to_id, hyper_params.word_cutoff);
    cout << "completed" << endl;

    if (default_config.program_mode == ProgramMode::INTERACTING) {
        hyper_params.beam_size = beam_size;
        interact(default_config, hyper_params, model_params, all_idf, word_counts,
                hyper_params.word_cutoff, black_list);
    } else if (default_config.program_mode == ProgramMode::DECODING) {
        globalPoolEnabled() = false;
        hyper_params.beam_size = beam_size;
//        decodeTestPosts(hyper_params, model_params, default_config, all_idf,
//                response_idf_info_list, test_post_and_responses, post_sentences,
//                response_sentences, black_list);
    } else if (default_config.program_mode == ProgramMode::METRIC) {
        const string &model_file_path = default_config.input_model_file;
        cout << format("model_file_path:%1%") % model_file_path << endl;
        ModelParams model_params;
        int epoch;
        loadModel(default_config, hyper_params, model_params, epoch, model_file_path,
                allocate_model_params);
        float perplex = metricTestPosts(hyper_params, model_params,
                test_post_and_responses, post_sentences, response_sentences,
                response_idf_info_list);
        cout << format("model %1% perplex is %2%") % model_file_path % perplex << endl;
    } else if (default_config.program_mode == ProgramMode::TRAINING) {
        insnet::AdamOptimizer optimizer(model_params.tunableParams(), hyper_params.learning_rate);

        CheckGrad grad_checker;
        if (default_config.check_grad) {
            grad_checker.init(model_params.tunableParams());
        }


        int iteration = 0;
        string last_saved_model;

        default_random_engine engine(default_config.seed);
        for (int epoch = epoch_begin + 1; epoch < default_config.max_epoch; ++epoch) {
            dtype loss_sum = 0.0f;
            int word_sum = 0;
            dtype keyword_loss_sum = 0;
            int keyword_sum = 0;
            dtype normal_loss_sum = 0;
            cout << "epoch:" << epoch << endl;
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


            for (int batch_i = 0; batch_i < batch_count +
                    (train_conversation_pairs.size() > hyper_params.batch_size * batch_count);
                    ++batch_i) {
                if (batch_i % 10 == 5) {
                    cout << format("batch_i:%1% iteration:%2%") % batch_i % iteration << endl;
                }
                optimizer.setLearningRate(hyper_params.learning_rate);

                if (batch_i % 10 == 5) {
                    cout << "lr:" << optimizer.getLearningRate() << endl;
                }
                int batch_size = batch_i == batch_count ?
                    train_conversation_pairs.size() % hyper_params.batch_size :
                    hyper_params.batch_size;
                Graph graph;
                vector<shared_ptr<GraphBuilder>> graph_builders;
                vector<ConversationPair> conversation_pair_in_batch;
                auto getSentenceIndex = [batch_i, batch_count](int i) {
                    return i * batch_count + batch_i;
                };
                vector<pair<vector<Node *>, vector<Node *>>> results;
                for (int i = 0; i < batch_size; ++i) {
                    shared_ptr<GraphBuilder> graph_builder(new GraphBuilder);
                    graph_builders.push_back(graph_builder);
                    int instance_index = getSentenceIndex(i);
                    int post_id = train_conversation_pairs.at(instance_index).post_id;
                    conversation_pair_in_batch.push_back(train_conversation_pairs.at(
                                instance_index));
                    auto post_sentence = post_sentences.at(post_id);
                    graph_builder->forward(graph, post_sentence, hyper_params, model_params);
                    int response_id = train_conversation_pairs.at(instance_index).response_id;
                    auto response_sentence = response_sentences.at(response_id);
                    const WordIdfInfo &idf_info = response_idf_info_list.at(response_id);
                    auto p = graph_builder->forwardDecoder(graph, *graph_builder->encoder_hiddens,
                            response_sentence,
                            idf_info.keywords_behind, hyper_params, model_params);
                    results.push_back(move(p));
                    word_sum += response_sentence.size();
                }

                graph.forward();

                for (int i = 0; i < batch_size; ++i) {
                    int instance_index = getSentenceIndex(i);
                    int response_id = train_conversation_pairs.at(instance_index).response_id;
                    auto response_sentence = response_sentences.at(response_id);
                    vector<int> word_ids = toIds(response_sentence, model_params.lookup_table);
                    for (int j = 0; j < word_ids.size(); ++j) {
                        if (j < word_ids.size() - 1) {
                            --word_ids.at(j);
                        } else {
                            if (word_ids.at(j) != 0) {
                                cerr << "stop symbol id is not 0:" << word_ids.at(j) << endl;
                                abort();
                            }
                        }
                    }
                    vector<Node*> result_nodes = results.at(i).second;
                    dtype result;
                    try {
                        result = MaxLogProbabilityLossWithInconsistentDims(result_nodes,
                                word_ids, model_params.lookup_table.nVSize);
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
                            model_params.lookup_table.vocab.from_string(word) %
                            model_params.lookup_table.vocab.from_string(keyword) << endl;

                        print(response_sentence);
                        print(response_idf_info_list.at(response_id).keywords_behind);
                        abort();
                    }
                    loss_sum += result;
                    normal_loss_sum += result;
                    const WordIdfInfo &response_idf = response_idf_info_list.at(response_id);
                    auto keyword_nodes_and_ids = keywordNodesAndIds(
                            results.at(i).first, response_idf, model_params);
                    auto keyword_result = MaxLogProbabilityLossWithInconsistentDims(
                            keyword_nodes_and_ids.first, keyword_nodes_and_ids.second,
                            model_params.lookup_table.size());
                    keyword_sum += keyword_nodes_and_ids.second.size();
                    loss_sum += keyword_result;
                    keyword_loss_sum += keyword_result;

                    if (batch_i % 10 == 5 && i == 0) {
                        int post_id = train_conversation_pairs.at(instance_index).post_id;
                        cout << "post:" << post_id << endl;
                        print(post_sentences.at(post_id));

                        cout << "golden answer:" << endl;
                        printWordIds(word_ids, model_params.lookup_table);

                        cout << "golden keywords:" << endl;
                        printWordIds(keyword_nodes_and_ids.second, model_params.lookup_table);
                    }
                }

                if (batch_i % 10 == 5) {
                    cout << "loss:" << loss_sum <<" ppl:" << std::exp(loss_sum / word_sum) <<
                        "normal ppl:" << std::exp(normal_loss_sum / word_sum) <<
                        "keyword ppl:" << std::exp(keyword_loss_sum / keyword_sum)<< endl;
                }

                graph.backward();

                if (default_config.check_grad) {
//                    auto loss_function = [&](const ConversationPair &conversation_pair) -> dtype {
//                        GraphBuilder graph_builder;
//                        Graph graph;

//                        graph_builder.forward(graph, post_sentences.at(conversation_pair.post_id),
//                                hyper_params, model_params, true);

//                        DecoderComponents decoder_components;
//                        graph_builder.forwardDecoder(graph, decoder_components,
//                                response_sentences.at(conversation_pair.response_id),
//                                response_idf_info_list.at(
//                                    conversation_pair.response_id).keywords_behind,
//                                hyper_params, model_params, true);

//                        graph.compute();

//                        vector<int> word_ids = toIds(response_sentences.at(
//                                    conversation_pair.response_id), model_params.lookup_table);
//                        vector<Node*> result_nodes = toNodePointers(
//                                decoder_components.wordvector_to_onehots);
//                        const WordIdfInfo &response_idf = response_idf_info_list.at(
//                                conversation_pair.response_id);
//                        auto keyword_nodes_and_ids = keywordNodesAndIds(
//                                decoder_components, response_idf, model_params);
//                        return MaxLogProbabilityLossWithInconsistentDims(
//                                keyword_nodes_and_ids.first, keyword_nodes_and_ids.second, 1,
//                                model_params.lookup_table.nVSize).first +
//                            MaxLogProbabilityLossWithInconsistentDims( result_nodes, word_ids, 1,
//                                    model_params.lookup_table.nVSize).first;
//                    };
//                    cout << format("checking grad - conversation_pair size:%1%") %
//                        conversation_pair_in_batch.size() << endl;
//                    grad_checker.check<ConversationPair>(loss_function, conversation_pair_in_batch,
//                            "");
                }

                optimizer.step(10.0f);

                if (default_config.save_model_per_batch) {
                    saveModel(hyper_params, model_params, default_config.output_model_file_prefix,
                            epoch);
                }

                ++iteration;
            }

            float ppl = metricTestPosts(hyper_params, model_params, dev_post_and_responses,
                    post_sentences, response_sentences, response_idf_info_list);
            cout << fmt::format("dev ppl is {}", ppl) << endl;

            cout << "loss_sum:" << loss_sum << endl;
            last_saved_model = saveModel(hyper_params, model_params,
                        default_config.output_model_file_prefix, epoch);
        }
    } else {
        abort();
    }

    return 0;
}
