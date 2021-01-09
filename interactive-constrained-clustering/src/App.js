import './App.css';
import React, { Component } from 'react';
import axios from 'axios';
import Router from './components/router'
import 'bootstrap/dist/css/bootstrap.min.css';
import { trackPromise } from 'react-promise-tracker';
import { Stats, PythonOutput } from './Python'

export const AppContext = React.createContext({
  dataArr: null,
  iterationCount: null,
  formInput: null,
  inputVerified: null,
  pythonPass: null,
  error: null,
  errorMessage: "",
  output: PythonOutput,
  stats: Stats,
  saveData: () => { },
  trackPython: () => { },
  saveForm: () => { },
  verifiedInput: () => { },
  pythonRestart: () => { }
});

class App extends Component {

  constructor(props) {
    super(props);
    this.state = {
      dataArr: null,
      iterationCount: 0, //default = 0
      formInput: null,
      inputVerified: false,
      pythonPass: true,
      error: false,
      errorMessage: "",
      output: "",
      stats: "",
      saveData: this.saveData,
      trackPython: this.trackPython,
      saveForm: this.saveForm,
      verifiedInput: this.verifiedInput,
      pythonRestart: this.pythonRestart
    };
  }

  trackPython = (ml, cl, unknown) => {
    trackPromise(
      this.runPython(ml, cl, unknown)
    )
  }

  runPython = (ml, cl, unknown) => {
    const promise = new Promise((resolve) => {
      const formData = new FormData();
      var baseUrl;
      if (window.location.origin === "http://localhost:3000") {
        baseUrl = "http://localhost:4500"
      } else {
        baseUrl = window.location.origin
      }
      console.log(baseUrl)
      formData.append('filename', this.state.formInput.filename)
      formData.append('interation_num', this.state.iterationCount + 1);
      this.setState({ iterationCount: this.state.iterationCount + 1 })
      formData.append('question_num', this.state.formInput.questionsPerIteration)
      formData.append('cluster_num', this.state.formInput.numberOfClusters)
      let totalML = this.state.formInput.ml.concat(ml)
      let totalCL = this.state.formInput.cl.concat(cl)
      let unknownC = this.state.formInput.unknown.concat(unknown)
      formData.append('ml', totalML)
      formData.append('cl', totalCL)
      formData.append('unknown', unknownC)
      this.setState({
        formInput: {
          ...this.state.formInput,
          ml: totalML,
          cl: totalCL,
          unknown: unknownC
        }
      });
      let outputsFromPython
      resolve(
        axios.post(baseUrl + '/python', formData, {
        }).then(res => {
          outputsFromPython = res.data.name
          var formState = this.state.formInput
          var outputArr = outputsFromPython.split("SEPERATOR")
          //The if catches any errors that Python may return. 
          if (parseInt(outputsFromPython) === 2) {
            this.handleMissingDataErrors("There was a constraint conflict. The tool can no longer improve.")
          } else if (parseInt(outputArr[3]) === 3) {
            this.handleMissingDataErrors("Due to the chosen constraints, the tool was unable to find " + this.state.formInput.questionsPerIteration + " questions. The tool can no longer improve.")
          } else {
            //Passed all the errors, continue on with the process. 
            this.setState({ stats: new Stats(formState.cl.length, formState.ml.length, formState.unknown.length, formState.maxConstraintPercent, this.state.dataArr.data.length, outputArr[1], outputArr[2], outputArr[0]) })
            this.setState({ output: new PythonOutput(outputArr[3].trim()) })
            this.errorFalse() //Remove any errors that may have been created previously. 
          }
        }).catch(_ => {
          if (parseInt(outputsFromPython) === 1) {
            this.setState({ errorMessage: "The dataset that was uploaded had categorical information. The tool can only handle numbers at this time." })
            this.setState({ inputVerified: false })
          }
        })
      )
    });
    return promise;
  }

  handleMissingDataErrors = (message) => {
    this.setState({ iterationCount: this.state.iterationCount - 1 })
    this.setState({ pythonPass: false })
    this.setState({ error: true })
    this.setState({ errorMessage: message })
  }

  saveData = (e) => {
    this.setState({ dataArr: e })
  }

  verifiedInput = () => {
    this.setState({ inputVerified: true })
  }

  pythonRestart = () => {
    this.setState({ iterationCount: 0 })
    this.setState({ error: true })
    this.setState({ pythonPass: true })
  }

  errorFalse = () => {
    this.setState({ error: false })
  }

  saveForm = (e) => {
    this.setState({ formInput: e })
  }

  render() {
    return (
      <div className="App">
        <AppContext.Provider value={this.state}>
          <Router></Router>
        </AppContext.Provider>
      </div>
    );
  }
}

export default App;
