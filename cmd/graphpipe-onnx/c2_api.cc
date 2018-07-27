#include "caffe2/core/flags.h"
#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/onnx/backend.h"
#include "caffe2/core/context_gpu.h"

#include "c2_api.h"

struct c2_engine_ctx
{
    int use_cuda;

    int batch_size;
    std::vector<std::string> all_inputs;
    std::map<int, std::string> inputs;
    std::map<int, std::string> outputs;
    caffe2::Workspace workspace;

    caffe2::onnx::Caffe2BackendRep *onnx_backend;
    caffe2::onnx::Caffe2Backend onnx_instance;
    std::map<std::string, std::vector<int64_t>> dims;
    std::map<std::string, int64_t> itemsizes;
    std::map<std::string, int64_t> rowsizes;
    std::map<std::string, int64_t> dtypes;

    caffe2::NetDef init_net, pred_net;

};


int _get_rowsize(c2_engine_ctx *ctx, char *name)
{
    std::map<std::string, int64_t>::iterator it;
    it = ctx->rowsizes.find(name);
    assert(it != ctx->rowsizes.end());
    return it->second;
}


int _get_itemsize(c2_engine_ctx *ctx, char *name)
{
    std::map<std::string, int64_t>::iterator it;
    it = ctx->itemsizes.find(name);
    assert(it != ctx->itemsizes.end());
    return it->second;
}

int c2_engine_get_itemsize(c2_engine_ctx *ctx, char *name) {
    return _get_itemsize(ctx, name);
}

int _get_dtype(c2_engine_ctx *ctx, char *name)
{
    std::map<std::string, int64_t>::iterator it;
    it = ctx->dtypes.find(name);
    assert(it != ctx->itemsizes.end());
    return it->second;
}

int _get_dtype(c2_engine_ctx *ctx, std::string name)
{
    std::map<std::string, int64_t>::iterator it;
    it = ctx->dtypes.find(name);
    assert(it != ctx->itemsizes.end());
    return it->second;
}

int c2_engine_get_dtype(c2_engine_ctx *ctx, char *name) {
    return _get_dtype(ctx, name);
}


void _do_tensor_copy(c2_engine_ctx *ctx, caffe2::Blob *blob, caffe2::TensorCPU &input) {
    if (ctx->use_cuda) {
        auto t = blob->GetMutable<caffe2::TensorCUDA>();
        t->CopyFrom(input);
    }
    else {
        auto t = blob->GetMutable<caffe2::TensorCPU>();
        t->CopyFrom(input);
    }
}

int c2_set_input_batch(c2_engine_ctx *ctx, char *name, void *input, int byte_count)
{
    int itemcount = byte_count / _get_itemsize(ctx, name);
    caffe2::Predictor::TensorMap::iterator it;
    auto* blob = ctx->workspace.GetBlob(name);
    if (blob) {

        std::map<std::string, std::vector<int64_t>>::iterator jt;
        jt = ctx->dims.find(name);
        if (jt == ctx->dims.end()) {
            assert(0);
            return -1;
        }

        auto dims = jt->second;
        int itemsize = _get_itemsize(ctx, name);
        int rowsize = _get_rowsize(ctx, name);
        if (itemcount % rowsize != 0) {
            assert(0);
            return -1;
        }

        /*
        std::cout << byte_count << " the byte_count\n";
        std::cout << itemcount << " the shape\n";
        std::cout << rowsize << " the rowsize\n";
        std::cout << _get_dtype(ctx, name) << " the rowsize\n";
        */

        switch (_get_dtype(ctx, name))
        {
#define COPY_INPUT(t) { t *arr = (t *) input; std::vector<t> batch_data {arr, arr + itemcount}; caffe2::TensorCPU input({(itemcount/rowsize), dims[1], dims[2], dims[3]}, batch_data, NULL); _do_tensor_copy(ctx, blob, input); }
            case caffe2::TensorProto_DataType_FLOAT:
                COPY_INPUT(float);
                break;
            case caffe2::TensorProto_DataType_FLOAT16:
                COPY_INPUT(caffe2::float16);
                break;
            case caffe2::TensorProto_DataType_INT32:
                COPY_INPUT(int32_t);
                break;
            case caffe2::TensorProto_DataType_BYTE:
                COPY_INPUT(unsigned char);
                break;
            case caffe2::TensorProto_DataType_UINT8:
                COPY_INPUT(uint8_t);
                break;
            case caffe2::TensorProto_DataType_INT8:
                COPY_INPUT(int8_t);
                break;
            case caffe2::TensorProto_DataType_UINT16:
                COPY_INPUT(uint16_t);
                break;
            case caffe2::TensorProto_DataType_INT16:
                COPY_INPUT(int16_t);
                break;
            case caffe2::TensorProto_DataType_INT64:
                COPY_INPUT(int64_t);
                break;
            case caffe2::TensorProto_DataType_DOUBLE:
                COPY_INPUT(double);
                break;
            default:
                return -1;
        }

    }
    return 0;
}

int c2_execute_batch(c2_engine_ctx *ctx) {
    ctx->workspace.RunNet(ctx->pred_net.name());
}

int c2_engine_get_input_count(c2_engine_ctx * ctx)
{
    return ctx->inputs.size();
}

const char *c2_engine_get_input_name(c2_engine_ctx * ctx, int i)
{
    return ctx->inputs[i].c_str();
}

int c2_engine_get_output_count(c2_engine_ctx * ctx)
{
    return ctx->outputs.size();
}

const char *c2_engine_get_output_name(c2_engine_ctx * ctx, int i)
{
    return ctx->outputs[i].c_str();
}

int c2_engine_get_output_size(c2_engine_ctx *ctx, int i)
{
    std::map<int, std::string>::iterator it;
    it = ctx->outputs.find(i);
    assert(it != ctx->outputs.end());
    auto* blob = ctx->workspace.GetBlob(it->second);
    assert(blob);

    if (ctx->use_cuda) {
        auto &t = blob->Get<caffe2::TensorCUDA>();
        size_t size = t.size() * t.itemsize();
        return size;
    }
    else {
        auto &t = blob->Get<caffe2::TensorCPU>();
        size_t size = t.size() * t.itemsize();
        return size;
    }
}

int c2_engine_get_output(c2_engine_ctx *ctx, int i, void *output)
{
    std::map<int, std::string>::iterator it;
    it = ctx->outputs.find(i);
    assert(it != ctx->outputs.end());
    auto* blob = ctx->workspace.GetBlob(it->second);
    assert(blob);

    const void *d;
    size_t size;
    if (ctx->use_cuda) {
        auto t = caffe2::TensorCPU(blob->Get<caffe2::TensorCUDA>());
        size = t.size() * t.itemsize();
        d = t.raw_data();
        memcpy(output, d, size);
    }
    else {
        auto &t = blob->Get<caffe2::TensorCPU>();
        size = t.size() * t.itemsize();
        d = t.raw_data();
        memcpy(output, d, size);
    }

    return size;
}

int c2_engine_get_output_index(c2_engine_ctx *ctx, char *name) {
    std::map<int, std::string>::iterator it;
    for (it=ctx->outputs.begin();it!=ctx->outputs.end();it++) {
        if (it->second.compare(name) == 0) {
            return it->first;
        }
    }
    return -1;
}

void c2_engine_register_input(c2_engine_ctx *ctx, char *name, int64_t *shape, int len, int dtype)
{
    std::vector<int64_t> tmp;
	for (int i=0; i < len; i++) {
        tmp.push_back(shape[i]);
	}
    ctx->dims[name] = tmp;
    ctx->inputs[ctx->inputs.size()] = name;
    ctx->dtypes[name] = dtype;
}

int c2_engine_get_dimensions(c2_engine_ctx *ctx, char *name, int64_t *dimensions) {
    std::map<std::string, std::vector<int64_t>>::iterator it;
    it = ctx->dims.find(name);
	if (it != ctx->dims.end()) {
		for (int i=0; i<it->second.size(); i++) {
      		dimensions[i] = it->second[i];
		}
		return it->second.size();
	}
	return -1;
}

c2_engine_ctx* c2_engine_create(int use_cuda) {
	c2_engine_ctx *ctx = new c2_engine_ctx();
    ctx->use_cuda = use_cuda;
    return ctx;
}

int _initialize(c2_engine_ctx *ctx) {
    if (ctx->use_cuda) {
        ctx->init_net.mutable_device_option()->set_device_type(caffe2::DeviceType::CUDA);
        ctx->pred_net.mutable_device_option()->set_device_type(caffe2::DeviceType::CUDA);
    }
    else {
        ctx->pred_net.mutable_device_option()->set_device_type(caffe2::DeviceType::CPU);
        ctx->init_net.mutable_device_option()->set_device_type(caffe2::DeviceType::CPU);
        for(int i = 0; i<ctx->pred_net.op_size(); ++i) {
            ctx->pred_net.mutable_op(i)->mutable_device_option()->set_device_type(caffe2::DeviceType::CPU);
        }
        for(int i = 0; i < ctx->init_net.op_size(); ++i){
            ctx->init_net.mutable_op(i)->mutable_device_option()->set_device_type(caffe2::DeviceType::CPU);
        }
    }

    CAFFE_ENFORCE(ctx->workspace.RunNetOnce(ctx->init_net));
    for (int i=0; i<ctx->pred_net.external_output().size(); i++) {
        ctx->outputs[i] = ctx->pred_net.external_output(i);
        std::vector<int64_t> tmp;
        ctx->dims[ctx->pred_net.external_output(i)] = tmp;
    }

    for (int i=0; i<ctx->pred_net.external_input().size(); i++) {
        ctx->all_inputs.push_back(ctx->pred_net.external_input(i));
        auto* blob = ctx->workspace.GetBlob(ctx->pred_net.external_input(i));
        if (blob) {
#if FALSE
            if (ctx->use_cuda) {
                auto data = caffe2::TensorCPU(blob->Get<caffe2::TensorCUDA>());
                ctx->dtypes[ctx->pred_net.external_input(i)] = caffe2::TypeMetaToDataType(data.meta());
            }
            else {
                auto &data = blob->Get<caffe2::TensorCPU>();
                std::cout << data.DebugString() << "\n";
                std::cout << data.meta().name() << "\n";
                /*
                std::cout << data.meta().id() << " ok\n";
                std::cout << caffe2::gTypeNames << " ok\n";
                std::cout << caffe2::TypeMetaToDataType(data.meta()) << "\n";
                */
                ctx->dtypes[ctx->pred_net.external_input(i)] = caffe2::TypeMetaToDataType(data.meta());
            }
#endif
        } else {
            ctx->workspace.CreateBlob(ctx->pred_net.external_input(i));
        }
    }
    CAFFE_ENFORCE(ctx->workspace.CreateNet(ctx->pred_net));


    std::map<int, std::string>::iterator it;
    for (it=ctx->inputs.begin(); it != ctx->inputs.end(); it++) {
        std::map<std::string, std::vector<int64_t>>::iterator jt;
        jt = ctx->dims.find(it->second);
        if (jt == ctx->dims.end()) {
            assert(0);
        }
        if (std::find(ctx->all_inputs.begin(), ctx->all_inputs.end(), it->second)==ctx->all_inputs.end()) {
            std::cerr << "Specified value input not found in graph: " << it->second << "\n";
            std::cerr << "Aborting\n";
            return -1;
        }

        std::vector<int64_t> dims = jt->second;
        size_t size = 1;
        for (int i=1;i<dims.size();i++) {
          size *= dims[i];
        }
        auto *blob = ctx->workspace.GetBlob(it->second);
        int dtype = _get_dtype(ctx, it->second);
        switch (dtype)
        {
#define SETUP_INPUT(t) { std::vector<t> test_data(size); caffe2::TensorCPU input({1, dims[1], dims[2], dims[3]}, test_data, NULL); _do_tensor_copy(ctx, blob, input); ctx->itemsizes[it->second] = input.itemsize(); }
            case caffe2::TensorProto_DataType_FLOAT:
                SETUP_INPUT(float)
                break;
            case caffe2::TensorProto_DataType_FLOAT16:
                SETUP_INPUT(caffe2::float16)
                break;
            case caffe2::TensorProto_DataType_INT32:
                SETUP_INPUT(int32_t)
                break;
            case caffe2::TensorProto_DataType_BYTE:
                SETUP_INPUT(unsigned char)
                break;
            case caffe2::TensorProto_DataType_UINT8:
                SETUP_INPUT(uint8_t)
                break;
            case caffe2::TensorProto_DataType_INT8:
                SETUP_INPUT(int8_t)
                break;
            case caffe2::TensorProto_DataType_UINT16:
                SETUP_INPUT(uint16_t)
                break;
            case caffe2::TensorProto_DataType_INT16:
                SETUP_INPUT(int16_t)
                break;
            case caffe2::TensorProto_DataType_INT64:
                SETUP_INPUT(int64_t)
                break;
            case caffe2::TensorProto_DataType_DOUBLE:
                SETUP_INPUT(double)
                break;
                
            default:
                return -1;
        }
        ctx->rowsizes[it->second] = size;
    }

    ctx->workspace.RunNet(ctx->pred_net.name());

    for (it=ctx->outputs.begin(); it!=ctx->outputs.end(); it++) {
        auto* blob = ctx->workspace.GetBlob(it->second);

        if (ctx->use_cuda) {
            auto data = caffe2::TensorCPU(blob->Get<caffe2::TensorCUDA>());
            std::vector<int64_t> tmp;
            for (int j=0;j<data.dims().size(); j++) {
                tmp.push_back(data.dims()[j]);
            }
            ctx->dims[it->second] = tmp;
            ctx->itemsizes[it->second] = data.itemsize();
            ctx->dtypes[it->second] = caffe2::TypeMetaToDataType(data.meta());
        }
        else {
            auto &data = blob->Get<caffe2::TensorCPU>();
            std::vector<int64_t> tmp;
            for (int j=0;j<data.dims().size(); j++) {
                tmp.push_back(data.dims()[j]);
            }
            ctx->dims[it->second] = tmp;
            ctx->itemsizes[it->second] = data.itemsize();
            ctx->dtypes[it->second] = caffe2::TypeMetaToDataType(data.meta());
        }

    }
	return 0;
}

int c2_engine_initialize_caffe2(c2_engine_ctx *ctx, char *init_data, size_t init_data_len, char *pred_data, size_t pred_data_len) {
    std::vector<std::string> strings {"ignore", "--caffe2_omp_num_threads", "8"};
    std::vector<char*> cstrings;
    for(size_t i = 0; i < strings.size(); ++i) {
        cstrings.push_back(const_cast<char*>(strings[i].c_str()));
    }

	int _argc = cstrings.size();
	char **_argv = &cstrings[0];


    caffe2::GlobalInit(&_argc, &_argv);

    std::string pred_content(pred_data, pred_data_len);
    std::string init_content(init_data, init_data_len);

    CAFFE_ENFORCE(caffe2::ParseProtoFromLargeString(init_content, &ctx->init_net));
    CAFFE_ENFORCE(caffe2::ParseProtoFromLargeString(pred_content, &ctx->pred_net));
    return _initialize(ctx); 
}

int c2_engine_initialize_onnx(c2_engine_ctx *ctx, char *model_data, size_t model_data_len) {
    std::vector<std::string> strings {"ignore", "--caffe2_omp_num_threads", "8"};
    std::vector<char*> cstrings;
    for(size_t i = 0; i < strings.size(); ++i) {
        cstrings.push_back(const_cast<char*>(strings[i].c_str()));
    }

	int _argc = cstrings.size();
	char **_argv = &cstrings[0];

    caffe2::GlobalInit(&_argc, &_argv);
    std::vector<caffe2::onnx::Caffe2Ops> extras;

    std::string content(model_data, model_data_len);
    ctx->onnx_backend = ctx->onnx_instance.Prepare(content, (ctx->use_cuda ? "CUDA" : "CPU"), extras);
    ctx->init_net = ctx->onnx_backend->init_net();
    ctx->pred_net = ctx->onnx_backend->pred_net();
    return _initialize(ctx); 
}

