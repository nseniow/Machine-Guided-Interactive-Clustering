import React from 'react';
import { useField } from "formik";

/**
 * Checkbox input for basic inputs.
 * @param props 
 * name: used as for the use of the label for the input
 * label: the text used within the label. 
 */
export const MyCheckBoxInput = (props) => {
    return (
        <div className="input-group input-group-append rounded-right ">
            <div>
                <label style={{marginRight: '1em'}}>
                    <input type="checkbox" class="radio" id="INNE" value="INNE" checked="true"/> iNNE  </label>
                <label style={{marginRight: '1em'}}>
                    <input type="checkbox" class="radio" id="COPOD" value="COPOD" checked="true"/> COPOD  </label>
                <label style={{marginRight: '1em'}}>
                    <input type="checkbox" class="radio" id="IF" value="IF" checked="true"/> Isolation Forest  </label>
                <label style={{marginRight: '1em'}}>
                    <input type="checkbox" class="radio" id="LOF" value="LOF" checked="true"/> LOF  </label>
                <label style={{marginRight: '1em'}}>
                    <input type="checkbox" class="radio" id="SIL" value="SIL" checked="true"/> Sihlouette  </label>
            </div>
        </div>            
    );
};
