import React from 'react';
import { Field } from "formik";

/**
 * Checkbox input for basic inputs.
 * @param props 
 * name: used as for the use of the label for the input
 * label: the text used within the label. 
 */
export const MyCheckBoxInput = (props) => {
    return (
        <div className="input-group input-group-append rounded-right ">
            <div role="group">
                <label style={{marginRight: '1em'}}>
                    <Field name="checked" type="checkbox" class="radio" id="INNE" value="INNE"/> iNNE  </label>
                <label style={{marginRight: '1em'}}>
                    <Field name="checked"  type="checkbox" class="radio" id="COPOD" value="COPOD"/> COPOD  </label>
                <label style={{marginRight: '1em'}}>
                    <Field name="checked"  type="checkbox" class="radio" id="IF" value="IF"/> Isolation Forest  </label>
                <label style={{marginRight: '1em'}}>
                    <Field name="checked"  type="checkbox" class="radio" id="LOF" value="LOF"/> LOF  </label>
                <label style={{marginRight: '1em'}}>
                    <Field name="checked"  type="checkbox" class="radio" id="SIL" value="SIL"/> Silhouette  </label>
            </div>
        </div>            
    );
};
