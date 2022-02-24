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
                    <input type="checkbox" class="radio" id="INNE" value="INNE" /> iNNE  </label>
                <label style={{marginRight: '1em'}}>
                    <input type="checkbox" class="radio" id="ABOD" value="ABOD" /> ABOD  </label>
                <label style={{marginRight: '1em'}}>
                    <input type="checkbox" class="radio" id="IF" value="IF" /> Isolation Forest  </label>
                <label style={{marginRight: '1em'}}>
                    <input type="checkbox" class="radio" id="LOF" value="LOF" /> LOF  </label>
                <label style={{marginRight: '1em'}}>
                    <input type="checkbox" class="radio" id="SIL" value="SIL" /> Sihlouette  </label>
            </div>
        </div>            
    );
};
