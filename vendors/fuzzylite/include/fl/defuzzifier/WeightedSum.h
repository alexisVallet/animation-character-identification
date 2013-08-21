/*   Copyright 2013 Juan Rada-Vilela

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

/* 
 * File:   WeightedSum.h
 * Author: jcrada
 *
 * Created on 8 May 2013, 4:15 PM
 */

#ifndef FL_WEIGHTEDSUM_H
#define	FL_WEIGHTEDSUM_H


#include "fl/defuzzifier/Defuzzifier.h"

namespace fl {

    class FL_EXPORT WeightedSum : public Defuzzifier {
    public:

        WeightedSum(int divisions = FL_DIVISIONS);
        virtual ~WeightedSum();

        virtual std::string className() const ;
        virtual scalar defuzzify(const Term* term,
                scalar minimum, scalar maximum) const;

    };
}

#endif	/* FL_WEIGHTEDSUM_H */

