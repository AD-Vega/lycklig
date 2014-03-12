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
  bool parse(int argc, char *argv[]);

  unsigned int boxsize = 60;
  float val_threshold = 0.3;
  float surf_threshold = 0.4;
  unsigned int maxmove = 20;
  std::string output_file;
  std::vector<std::string> files;
};

#endif // REGISTRATIONPARAMS_H
