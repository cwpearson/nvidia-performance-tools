#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace argparse {

class OptionBase {
public:
  virtual void set_val(const std::string &valStr) = 0;
  virtual const std::string &long_str() = 0;
};

template <typename T> class Option : public OptionBase {
  std::string long_;
  T *val_;

public:
  Option(T &val, const std::string &l) : long_(l), val_(&val) {}
  void set_val(const std::string &val) override { set_val((T *)nullptr, val); }
  const std::string &long_str() override { return long_; }

private:
  void set_val(size_t *, const std::string &val) { // convert to size_t
    *val_ = std::stoull(val);
  }
  void set_val(double *, const std::string &val) { // convert to double
    *val_ = std::stod(val);
  }
  void set_val(float *, const std::string &val) { // convert to float
    *val_ = std::stof(val);
  }
  void set_val(int *, const std::string &val) { // convert to int
    *val_ = std::stoi(val);
  }
  void set_val(std::string *, const std::string &val) { // convert to string
    *val_ = val;
  }
};

class Flag {
  std::string long_;
  std::string short_;
  std::string help_;
  bool *val_;

public:
  Flag(bool &val, const std::string &l, const std::string &s)
      : long_(l), short_(s), val_(&val) {}

  const std::string &long_str() const noexcept { return long_; }
  const std::string &short_str() const noexcept { return short_; }

  void set() const noexcept { *val_ = true; }

  void help(const std::string &s) { help_ = s; }

  const std::string &help_str() const noexcept { return help_; }
};

class PosnlBase {
public:
  virtual bool is_required() = 0;
  virtual PosnlBase *required() = 0;
  virtual void set_val(const std::string &val) = 0;
  virtual bool found() = 0;
};

template <typename T> class Positional : public PosnlBase {
  bool required_;
  T *val_;
  bool found_;

public:
  Positional(T &val) : required_(false), val_(&val), found_(false) {}

  PosnlBase *required() override {
    required_ = true;
    return this;
  }

  bool is_required() override { return required_; }

  // use nullpointer type to disambiguate call
  // https://stackoverflow.com/questions/5512910/explicit-specialization-of-template-class-member-function
  void set_val(const std::string &val) {
    found_ = true;
    set_val((T *)nullptr, val);
  }

  bool found() override { return found_; }

private:
  // https://stackoverflow.com/questions/5512910/explicit-specialization-of-template-class-member-function
  template <typename C>
  void get_as(C *, const std::string &val) { // to be overridden
  }
  void set_val(size_t *, const std::string &val) { // convert to size_t
    *val_ = std::stoull(val);
  }
  void set_val(double *, const std::string &val) { // convert to double
    *val_ = std::stod(val);
  }
  void set_val(float *, const std::string &val) { // convert to float
    *val_ = std::stof(val);
  }
  void set_val(int *, const std::string &val) { // convert to int
    *val_ = std::stoi(val);
  }
  void set_val(std::string *, const std::string &val) { // convert to string
    *val_ = val;
  }
};

class Parser {

  std::string description_;
  bool noUnrecognized_; // error on unrecognized flags / opts
  bool help_;           // help has been requested
  bool consume_;        // remove consumed values from argc, argv

  std::vector<OptionBase *> opts_;
  std::vector<Flag> flags_;
  std::vector<PosnlBase *> posnls_;

  static bool starts_with(const std::string &s, const std::string &prefix) {
    if (s.rfind(prefix, 0) == 0) {
      return true;
    }
    return false;
  }

  OptionBase *match_opt(const char *arg) const {
    std::string sarg(arg);
    for (int64_t i = int64_t(opts_.size()) - 1; i >= 0; --i) {
      if (opts_[i]->long_str() == sarg) {
        return opts_[i];
      }
    }
    return nullptr;
  }

  Flag *match_flag(const char *arg) {
    std::string sarg(arg);
    for (int64_t i = int64_t(flags_.size()) - 1; i >= 0; --i) {
      if (flags_[i].long_str() == sarg || flags_[i].short_str() == sarg) {
        return &flags_[i];
      }
    }
    return nullptr;
  }

public:
  Parser() : noUnrecognized_(false), help_(false), consume_(true) {
    add_flag(help_, "--help", "-h")->help("Print help message");
  }
  Parser(const std::string &description)
      : description_(description), noUnrecognized_(false), help_(false),
        consume_(true) {
    add_flag(help_, "--help", "-h")->help("Print help message");
  }

  bool parse(int &argc, char **argv) {

    std::vector<char *> newArgv;
    if (argc > 0) {
      newArgv.push_back(argv[0]);
    }

    size_t pi = 0;        // positional argument position
    bool optsOkay = true; // okay to interpret as opt/flag
    for (int i = 1; i < argc; ++i) {

      // try interpreting as a flag or option if it looks like one
      if (optsOkay && starts_with(argv[i], "-")) {
        // '--' indicates only positional arguments follow
        if (argv[i] == std::string("--")) {
          optsOkay = false;
          continue;
        }
        OptionBase *opt = match_opt(argv[i]);
        if (opt) {
          opt->set_val(argv[i + 1]);
          ++i;
          continue;
        }
        Flag *flag = match_flag(argv[i]);
        if (flag) {
          flag->set();
          continue;
        }
        newArgv.push_back(argv[i]);
        if (noUnrecognized_) {
          std::cerr << "unrecognized " << argv[i] << "\n";
          return false;
        }
      } else { // otherwise try it as positional
        if (pi < posnls_.size()) {
          posnls_[pi]->set_val(argv[i]);
          ++pi;
        } else {
          newArgv.push_back(argv[i]);
          std::cerr << "encountered unexpected positional argument " << pi
                    << ": " << argv[i] << "\n";
        }
      }
    }

    for (; pi < posnls_.size(); ++pi) {
      if (posnls_[pi]->is_required()) {
        std::cerr << "missing required positional argument " << pi << "\n";
        return false;
      }
    }

    if (consume_) {
      argc = newArgv.size();
      for (int i = 0; i < argc; ++i) {
        argv[i] = newArgv[i];
      }
    }

    return true;
  };

  template <typename T> void add_option(T &val, const std::string &l) {
    opts_.push_back(new Option<T>(val, l));
  }

  Flag *add_flag(bool &val, const std::string &l, const std::string &s = "") {
    flags_.push_back(Flag(val, l, s));
    return &(flags_.back());
  }

  template <typename T> PosnlBase *add_positional(T &val) {
    posnls_.push_back(new Positional<T>(val));
    return posnls_.back();
  }

  std::string help() const {
    std::stringstream ss;

    ss << description_ << "\n";

    for (auto &o : opts_) {
      ss << o->long_str() << "\n";
    }

    for (auto &f : flags_) {
      ss << "  " << f.short_str() << ", " << f.long_str();
      ss << "\t\t" << f.help_str();
      ss << "\n";
    }

    return ss.str();
  }

  /*! \brief error on unrecognized flags and options
   */
  void no_unrecognized() { noUnrecognized_ = true; }

  /*! \brief don't modify argc/argv
   */
  void no_consume() { consume_ = false; }

  bool need_help() const noexcept { return help_; }
};

} // namespace argparse