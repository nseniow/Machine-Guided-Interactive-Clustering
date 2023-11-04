import FileUpload from './inputForm/inputForm';
import React, { Component } from 'react';
import { Col } from 'react-bootstrap';
import Notification from '../../notification'
import { AppContext } from "../../../App"
import InfoPanel from './inputForm/infoPanel';

class Landing extends Component {
    constructor(props){
        super(props);
        this.state = {
            showInfo: false,
        }
    }

    changeShowInfo = () => {
        this.setState({showInfo: !this.state.showInfo});
    }

    render() {
        return (
            <>
                <AppContext.Consumer>
                    {context => (
                        <>
                            <Notification text={context.notifMessage} show={context.error} type=""/>
                            <div className="rowNoMargin imgSection">
                                {this.state.showInfo
                                ? 
                                <Col class="col-sm-6">
                                    <InfoPanel showInfo={this.changeShowInfo}/>
                                </Col>
                                : 
                                <Col className="col-sm-6 align-middle align-items-center text-center leftHalf">
                                    <h1 className="text-white titleFontSize">Machine Guided Interactive Clustering (MAGIC)</h1>
                                </Col>
                                }   
                                <Col className="col-sm-6 align-middle align-items-center">
                                    <FileUpload showInfo={this.changeShowInfo}/>
                                </Col>
                            </div>
                        </>
                    )}
                </AppContext.Consumer>
            </>
        );
    }
}

export default Landing;