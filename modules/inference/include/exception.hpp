
#ifndef MODULES_INFERENCE_EXCEPTION_HPP_
#define MODULES_INFERENCE_EXCEPTION_HPP_


#include <stdexcept>
#include <string>

/**
 * Registor exception class derived from CnstreamError
 *
 * @c CNAME Class name.
 */
#define CNSTREAM_REGISTER_EXCEPTION(CNAME)                                     \
  class CNAME##Error : public std::runtime_error {                             \
   public:                                                                     \
    explicit CNAME##Error(const std::string &msg) : std::runtime_error(msg) {} \
  };

#endif