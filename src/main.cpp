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

#include <iostream>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <omp.h>
#include "imageops.h"
#include "globalregistrator.h"
#include "rbfwarper.h"
#include "registrationparams.h"
#include "registrationcontext.h"

using namespace cv;

int main(const int argc, const char *argv[]) {
  // Parse command line parameters.
  registrationParams params;
  if (!params.parse(argc, argv))
    return 1;

  // This should be called in order for OpenMP parallelization to work with
  // ImageMagick.
  Magick::InitializeMagick(NULL);

  registrationContext context;

  // Load a state file if one was supplied.
  if (!params.read_state_file.empty()) {
    std::cerr << "Reading state from '" << params.read_state_file << "'\n";
    FileStorage readStateFS(params.read_state_file, FileStorage::READ);
    context = registrationContext(readStateFS);
    std::cerr << context.images().size() << " input files read from state file\n";
  }
  else {
    // No state file - we are starting from scratch. Initialize registration
    // context from command line parameters.
    context.boxsize(params.boxsize);
    std::vector<inputImage> images;
    for (auto& file : params.files)
      images.push_back(inputImage(file));
    context.images(images);
    std::cerr << context.images().size() << " input files listed on command line\n";
  }

  // preregistration stage
  if (params.stage_prereg) {
    if (params.prereg_img.empty())
      params.prereg_img = context.images().at(0).filename;
    Mat globalRefimg(grayReader().read(params.prereg_img));
    if (params.prereg_maxmove == 0) {
      params.prereg_maxmove = std::min(globalRefimg.rows, globalRefimg.cols)/2;
    }
    std::cerr << "Pre-registering\n";
    globalRegistrator::getGlobalShifts(params, context, globalRefimg, true);
  }

  // If we are going to do lucky imaging, we need a reference image. If there
  // was none in the state file, we need to make one now.
  if (params.stage_lucky && !context.refimgValid())
    params.stage_refimg = true;

  // reference image + possibly registration patches
  if (params.stage_refimg) {
    std::cerr << "Creating a stacked reference image\n";
    Mat rawRef = meanimg(params, context, true);

    if (params.only_refimg)
      imwrite(params.output_file, normalizeTo16Bits(rawRef));
    else {
      Mat refimg;
      rawRef.convertTo(refimg, CV_32F);
      cvtColor(refimg, refimg, CV_BGR2GRAY);
      context.refimg(refimg);

      // create registration patches
      std::cerr << "Lucky imaging: creating registration patches\n";
      auto patches = selectPointsHex(params, context);
      patches = filterPatchesByQuality(patches, context.refimg());
      context.patches(patches);
      std::cerr << context.patches().size() << " valid patches\n";

      // Changing the reference image invalidates lucky imaging shifts.
      context.clearShifts();
    }
  }

  if (params.stage_lucky || params.stage_stack) {
    if (params.stage_lucky && params.stage_stack)
      std::cerr << "Lucky imaging: registration, warping and stacking\n";
    else if (params.stage_lucky)
      std::cerr << "Lucky imaging: registration\n";
    else if (params.stage_stack && context.shiftsValid())
      std::cerr << "Stacking images (using data from lucky imaging)\n";
    else if (params.stage_stack)
      std::cerr << "Stacking images (no lucky imaging)\n";

    Mat finalsum = lucky(params, context, true);
    // Only save the result if there is something to save.
    if (params.stage_stack)
      imwrite(params.output_file, normalizeTo16Bits(finalsum));
  }

  if (!params.save_state_file.empty()) {
    std::cerr << "Saving state to '" << params.save_state_file << "'\n";
    FileStorage saveStateFS(params.save_state_file, FileStorage::WRITE);
    context.write(saveStateFS);
  }

  return 0;
}
