/*
 *    lycklig, image processing for lucky imaging.
 *    Copyright (C) 2013, 2014 Andrej Lajovic <andrej.lajovic@ad-vega.si>
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef REGISTRATIONPARAMS_H
#define REGISTRATIONPARAMS_H

#include <string>
#include <vector>

class registrationParams
{
public:
  bool parse(const int argc, const char *argv[]);

  bool stage_prereg = false;
  bool stage_refimg = false;
  bool stage_patches = false;
  bool stage_dedistort = false;
  bool stage_stack = false;

  // global registration
  enum class preregType { None, ExplicitImage, FirstImage, MiddleImage }
    prereg = preregType::None;
  std::string prereg_img;
  unsigned int prereg_maxmove = 0;

  // reference image + registration points
  bool only_refimg = false;
  bool crop = false;
  int boxsize = 60;
  bool boxsize_override = false;

  // dedistortion
  unsigned int maxmove = 20;

  // interpolation + stacking
  int supersampling = 1;

  // input options
  std::string read_state_file;
  std::vector<std::string> files;

  // output options
  std::string save_state_file;
  std::string output_file;
};

#endif // REGISTRATIONPARAMS_H
