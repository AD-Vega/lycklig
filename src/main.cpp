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
  registrationParams params;
  if (!params.parse(argc, argv))
    return 1;

  Magick::InitializeMagick(NULL);

  registrationContext context;
  context.images().reserve(params.files.size());
  for (auto& file : params.files)
    context.images().push_back(inputImage(file));

  std::cerr << context.images().size() << " input files listed\n";

  if (params.prereg) {
    Mat globalRefimg(grayReader().read(params.prereg_img));
    if (params.prereg_maxmove == 0) {
      params.prereg_maxmove = std::min(globalRefimg.rows, globalRefimg.cols)/2;
    }
    std::cerr << "Pre-registering\n";
    globalRegistrator::getGlobalShifts(params, context, globalRefimg, true);
  }

  std::cerr << "Creating a stacked reference image\n";
  Mat rawRef = meanimg(params, context, true);
  if (params.only_stack) {
    imwrite(params.output_file, normalizeTo16Bits(rawRef));
    return 0;
  }
  Mat refimg;
  rawRef.convertTo(refimg, CV_32F);
  cvtColor(refimg, refimg, CV_BGR2GRAY);
  context.refimg(refimg);

  std::cerr << "Lucky imaging: creating registration patches\n";
  auto patches = selectPointsHex(params, context);
  patches = filterPatchesByQuality(patches, context.refimg());
  context.patches(patches);
  std::cerr << context.patches().size() << " valid patches\n";

  std::cerr << "Lucky imaging: registration & warping\n";
  Mat finalsum = lucky(params, context, true);

  imwrite(params.output_file, normalizeTo16Bits(finalsum));
  return 0;
}
