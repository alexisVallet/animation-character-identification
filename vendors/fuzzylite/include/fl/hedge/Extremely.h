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
 * Extremely.h
 *
 *  Created on: 4/12/2012
 *      Author: jcrada
 */

#ifndef FL_EXTREMELY_H
#define FL_EXTREMELY_H

#include "fl/hedge/Hedge.h"

namespace fl {

    class FL_EXPORT Extremely : public Hedge {
    public:
        std::string name() const;
        scalar hedge(scalar x) const;
    };

} 
#endif /* FL_EXTREMELY_H */
