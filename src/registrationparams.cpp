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

#include <tclap/CmdLine.h>
#include "registrationparams.h"

class portionConstraint : public TCLAP::Constraint<float>
{
public:
  std::string description() const
    { return "between 0 and 1"; }
  std::string shortID() const
    { return "0.0-1.0"; }
  bool check(const float& value) const
    { return (value > 0.0) && (value < 1.0); }
} pConstraint;


template <typename num_t>
std::string defval(num_t d)
{
  return "(default " + std::to_string(d) + ")";
}


bool registrationParams::parse(int argc, char* argv[])
{
  try
  {
    TCLAP::CmdLine cmd("Registration of planetary images");
    TCLAP::ValueArg<unsigned int> arg_boxsize(
      "b", "boxsize", "Box size " + defval(boxsize), false, boxsize, "pixels");
    cmd.add(arg_boxsize);
    TCLAP::ValueArg<float> arg_val_threshold(
      "t", "val", "Value threshold " + defval(val_threshold), false, val_threshold, &pConstraint);
    cmd.add(arg_val_threshold);
    TCLAP::ValueArg<float> arg_surf_threshold(
      "s", "surf", "Surface threshold " + defval(surf_threshold), false, surf_threshold, &pConstraint);
    cmd.add(arg_surf_threshold);
    TCLAP::ValueArg<unsigned int> arg_maxmove(
      "m", "maxmove", "Maximum displacement " + defval(maxmove), false, maxmove, "pixels");
    cmd.add(arg_maxmove);
    TCLAP::ValueArg<std::string> arg_output_file(
      "o", "output", "Output file", true, "", "filename");
    cmd.add(arg_output_file);
    TCLAP::UnlabeledMultiArg<std::string> arg_files(
      "files", "Image files to process", true, "files");
    cmd.add(arg_files);
    
    cmd.parse(argc, argv);

    boxsize = arg_boxsize.getValue();
    val_threshold = arg_val_threshold.getValue();
    surf_threshold = arg_surf_threshold.getValue();
    maxmove = arg_maxmove.getValue();
    output_file = arg_output_file.getValue();
    files = arg_files.getValue();
  }
  catch (TCLAP::ArgException &e)
  {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    return false;
  }
  return true;
}
