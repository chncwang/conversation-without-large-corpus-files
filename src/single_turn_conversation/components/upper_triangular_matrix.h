#ifndef CONVERSATION_SRC_COM_UPPER_TRANGULAR_MATRIX
#define CONVERSATION_SRC_COM_UPPER_TRANGULAR_MATRIX
#include "N3LDG.h"

void initTriangularMatrixParam(UniParams &W, int size) {
    function<float(int, int)> cal_bound = [](int out, int in) {
        return static_cast<float>(sqrt(2.0 / in));
    };
    W.init(size, size, true, &cal_bound);

    for (int x = 0; x < size; ++x) {
        for (int y = x + 1; y < size; ++y) {
            W.W.val.mat()(y, x) = 0.0f;
        }
    }
    cout << W.W.val.mat() << endl;
#if USE_GPU
    W.W.val.copyFromHostToDevice();
#endif
}

class TriangularNode: public Node {
public:
    Node *in = nullptr;
    UniParams *param = nullptr;

    TriangularNode() : Node("triangular") {}

    void forward(Graph &graph, Node &innode) {
        in = &innode;
        in->addParent(this);
        graph.addNode(this);
    }

#if USE_GPU
    void compute() override {
    }

    void backward() override {
    }
#else
    void compute() override {
        int node_size = getDim();
        val().mat() = param->W.val.mat().block(0, 0, node_size, node_size) * in->getVal().mat();
        if (param->bUseB) {
            val().vec() += param->b.val.vec();
        }
    }

    void backward() override {
        auto grad = getLoss().mat() * in->getVal().mat().transpose();
        int node_size = getDim();
        for (int x = 0; x < node_size; ++x) {
            for (int y = 0; y < node_size; ++y) {
                param->W.grad.mat()(y, x) += grad(y, x);
            }
        }
        if (param->bUseB) {
            param->b.grad.vec() += getLoss().vec();
        }
        in->loss().mat() +=
            param->W.val.mat().block(0, 0, node_size, node_size).transpose() * getLoss().mat();
    }
#endif

    PExecutor generate() override;

    bool typeEqual(PNode other) override {
        TriangularNode *t = static_cast<TriangularNode*>(other);
        return Node::typeEqual(other) && param == t->param;
    }

    string typeSignature() const override {
        return Node::typeSignature() + "-" + addressToString(param);
    }
};

namespace n3ldg_plus {

Node *triangularNode(Graph &graph, Node &input, int dim, UniParams &params) {
    TriangularNode *tri = new TriangularNode;
    tri->init(dim);
    tri->param = &params;
    tri->forward(graph, input);
    return tri;
}

}

class TriangularExecutor : public Executor {};

PExecutor TriangularNode::generate() {
    return new TriangularExecutor;
}

#endif
