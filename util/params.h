#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

template <typename T>
struct default_value {
    default_value() : attr{} {}
    default_value(T t) : attr{t} {}
    T get_value() { return attr; }
    T attr;
};

struct option {
    option(const std::string &o, const std::string &s, std::string &v,
           const std::string &d)
        : op_long{o}, op_short{s}, dval{v}, desc{d} {};

    std::string op_long;
    std::string op_short;
    std::string dval;
    std::string desc;
};

struct flag {
    flag(const std::string &o, const std::string &s, const std::string &d)
        : flag_long{o}, flag_short{s}, desc{d} {};

    std::string flag_long;
    std::string flag_short;
    std::string desc;
};

struct param_val {
    param_val() = default;
    param_val(const std::string &s) : val{s} {}

    param_val(const param_val &rhs) { this->val = rhs.val; }
    param_val &operator=(const param_val &rhs) {
        this->val = rhs.val;
        return *this;
    }

    template <typename T>
    T as() {
        return static_cast<T>(val);
    }

   private:
    std::string val;
};

struct program_param {
    program_param(int argc, char *argv[])
        : exename{argv[0]}, tokens(argv + 1, argv + argc){};

    program_param &operator()(const char *op, std::string val,
                              std::string desc) {
        std::stringstream op_str(op);
        std::string op1{}, op2{};
        getline(op_str, op1, ',');
        if (op_str.good()) {
            getline(op_str, op2, ',');
        }
        if (op1.size() == 2) {
            options.push_back({op1, op2, val, desc});
        } else {
            options.push_back({op2, op1, val, desc});
        }
        param_val arg_val{val};
        if (!op1.empty()) option_list[op1] = arg_val;
        if (!op2.empty()) option_list[op2] = arg_val;

        return *this;
    }

    program_param &operator()(const char *flag, std::string desc) {
        std::stringstream flag_str(flag);
        std::string flag1{}, flag2{};
        getline(flag_str, flag1, ',');
        if (flag_str.good()) {
            getline(flag_str, flag2, ',');
        }
        if (flag1.size() == 2)
            flags.push_back({flag1, flag2, desc});
        else
            flags.push_back({flag2, flag1, desc});

        return *this;
    }

    void print_usage() {
        std::cout << exename << std::endl;
        for (auto &op : options) {
            std::cout << "\t" << op.op_long
                      << (op.op_long.empty() || op.op_short.empty() ? "" : "|")
                      << op.op_short << " [ Default: " << op.dval << "] "
                      << op.desc << std::endl;
        }
        for (auto &flag : flags) {
            std::cout << "\t"
                      << (flag.flag_long.empty() || flag.flag_short.empty()
                              ? ""
                              : "|")
                      << flag.flag_short << " [" << flag.desc << "]"
                      << std::endl;
        }
    }

    int parse() {
        for (int i = 0; i < tokens.size(); ++i) {
            auto token = tokens[i];

            auto op_iter = std::find_if(
                options.begin(), options.end(), [&token](option &o) {
                    return o.op_long == token || o.op_short == token;
                });
            if (op_iter != options.end()) {
                if (i == tokens.size() - 1) {
                    std::cout << "Option value isn't specied. ["
                              << op_iter->op_long << "|" << op_iter->op_short
                              << "]" << std::endl;
                    return program_param::NOK;
                }
                param_val arg_val{tokens[++i]};
                option_list[op_iter->op_long] = arg_val;
                option_list[op_iter->op_short] = arg_val;

                continue;
            }

            auto flag_iter =
                std::find_if(flags.begin(), flags.end(), [&token](flag &f) {
                    return f.flag_long == token || f.flag_short == token;
                });
            if (flag_iter != flags.end()) {
                param_val arg_val{{}};
                flag_list.insert(flag_iter->flag_long);
                flag_list.insert(flag_iter->flag_short);

                if (token == "-h" || token == "--help") {
                    return program_param::HELP;
                }
                continue;
            }

            std::cout << "Unknown flag: " << token << std::endl;
            print_usage();
            return program_param::NOK;
        }

        return program_param::OK;
    }

    template <typename T>
    T get_param(const std::string &op) {
        return option_list[op].as<T>();
    }

    enum { OK, NOK, HELP };

   private:
    std::string exename;
    std::unordered_map<std::string, param_val> option_list;
    std::unordered_set<std::string> flag_list;
    std::vector<option> options;
    std::vector<flag> flags;
    std::vector<std::string> tokens;
};
