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

using namespace cv;

int main(const int argc, const char *argv[]) {
  registrationParams params;
  if (!params.parse(argc, argv))
    return 1;

  Magick::InitializeMagick(NULL);

  fprintf(stderr, "%ld input files listed\n", params.files.size());

  globalRegistration globalRegResult;
  if (params.prereg) {
    Mat globalRefimg(grayReader().read(params.prereg_img));
    if (params.prereg_maxmove == 0) {
      params.prereg_maxmove = std::min(globalRefimg.rows, globalRefimg.cols)/2;
    }
    fprintf(stderr, "Pre-registering\n");
    globalRegResult = globalRegistrator::getGlobalShifts(globalRefimg, params, true);
  }

  fprintf(stderr, "Creating a stacked reference image\n");
  Mat rawRef = meanimg(params.files, globalRegResult, true);
  if (params.only_stack) {
    imwrite(params.output_file, normalizeTo16Bits(rawRef));
    return 0;
  }
  Mat refimg;
  rawRef.convertTo(refimg, CV_32F);
  cvtColor(refimg, refimg, CV_BGR2GRAY);

  fprintf(stderr, "Lucky imaging: creating registration patches\n");
  const unsigned int xydiff = params.boxsize/2;
  auto patches = selectPointsHex(refimg, params);
  patches = filterPatchesByQuality(patches, refimg);
  fprintf(stderr, "%ld valid patches\n", patches.size());
  rbfWarper rbf(patches, refimg.size(), xydiff/2);

  fprintf(stderr, "Lucky imaging: registration & warping\n");
  Mat finalsum = lucky(params, refimg, globalRegResult, patches, rbf, true);

  imwrite(params.output_file, normalizeTo16Bits(finalsum));
  return 0;
}
