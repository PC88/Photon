#ifndef _PPM_HPP_
#define _PPM_HPP_

/*
   A simple 24-bit Netpbm ASCII PPM "P3" loading and unloading class
   Copyright (c) 2016-2020 Paul Keir, University of the West of Scotland.
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <cassert>
#include <cerrno>

inline std::string get_file_contents(const char *);

struct ppm 
{

  void read(const char *filename, std::vector<unsigned char> &data)
  {
    std::string str = get_file_contents(filename);
    capacity = str.capacity();
    std::stringstream ss(str);
    ss >> magic >> w >> h >> max;
    assert(max <= std::numeric_limits<unsigned char>::max());
    data.reserve(w*h*nchannels);
    unsigned u;
    while (ss >> u)
      data.emplace_back(u); // Yes, pushing an uint into a vector of uchars
  }

  void write(const char *filename, const std::vector<unsigned char> &data)
  {
    const unsigned entries_per_line = 3;
    std::string str;
    str.reserve(capacity);
    std::stringstream ss(str);
    ss << magic << '\n' << w << ' ' << h << '\n' << max << '\n';
    unsigned count = 0;
    for (const unsigned col : data) { // Yes, reading a uchar as a uint
      ss << col << ' ';
      if (++count == entries_per_line) { ss << '\n'; count = 0; }
    }

    std::ofstream out(filename, std::ios::out);
    out << ss.rdbuf();
    out.close();
  }

  std::string magic;
  std::string::size_type capacity;
  unsigned w, h, max;
  const unsigned nchannels = 3;  // e.g. RGB; RGBA has 4 channels
};

// http://insanecoding.blogspot.co.uk/2011/11/how-to-read-in-file-in-c.html
inline std::string get_file_contents(const char *filename)
{
  std::ifstream in(filename, std::ios::in);
  if (in)
  {
    std::string contents;
    in.seekg(0, std::ios::end);
    contents.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(&contents[0], contents.size());
    in.close();
    return contents;
  }
  throw errno;
}

#endif // _PPM_HPP_
